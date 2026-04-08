import hashlib
import json
import os
import re
import time
 
import plotly.express as px
import streamlit as st
from groq import Groq
 
from invoice_agent import process_invoice
from invoice_parser import parse_file
from rag_engine import answer_question, index_invoice, suggest_questions

st.set_page_config(
    page_title="Invoice Reader Agent",
    page_icon="╮(╯▽╰)╭",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .flag-ok   { color: #0f6e56; background: #e1f5ee; padding: 2px 10px; border-radius: 12px; font-size: 12px; display: inline-block; margin: 2px; }
    .flag-err  { color: #993c1d; background: #faece7; padding: 2px 10px; border-radius: 12px; font-size: 12px; display: inline-block; margin: 2px; }
    .flag-warn { color: #854f0b; background: #faeeda; padding: 2px 10px; border-radius: 12px; font-size: 12px; display: inline-block; margin: 2px; }
    .sug-box   { background: #f5f5f3; border-radius: 8px; padding: 6px 12px; font-size: 13px; margin: 2px 0; }
</style>
""", unsafe_allow_html=True)


def _init():
    defaults = {
        "invoices":       [],
        "current_idx":    0,
        "chat_histories": {},
        "indexed_ids":    set(),
        "groq_client":    None,
        "pinecone_key":   None,
        "api_ready":      False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
 
_init()


def invoice_id(inv: dict) -> str:
    return hashlib.md5(
        f"{inv.get('_filename','')}{inv.get('invoiceNumber','')}{inv.get('total','')}".encode()
    ).hexdigest()[:12]
 
 
def currency_symbol(code: str) -> str:
    return {"USD": "$", "EUR": "€", "GBP": "£", "INR": "₹"}.get(code or "USD", (code or "$") + " ")
 
 
def fmt_money(amount, currency="USD"):
    if amount is None:
        return "--"
    sym = currency_symbol(currency)
    return f"{sym}{amount:,.2f}"
 
 
def urgency_label(u):
    return {"overdue": "[OVERDUE]", "critical": "[CRITICAL]", "high": "[HIGH]", "medium": "[MEDIUM]", "low": "[LOW]"}.get(u, "")


with st.sidebar:
    st.markdown("### API Keys")
    groq_key = st.text_input("Groq API Key", type="password",
                              value=os.getenv("GROQ_API_KEY", ""))
    pine_key = st.text_input("Pinecone API Key", type="password",
                              value=os.getenv("PINECONE_API_KEY", ""))
 
    if st.button("Connect", use_container_width=True):
        if groq_key and pine_key:
            try:
                st.session_state.groq_client  = Groq(api_key=groq_key)
                st.session_state.pinecone_key = pine_key
                st.session_state.api_ready    = True
                st.success("Connected.")
            except Exception as e:
                st.error(f"Connection failed: {e}")
        else:
            st.warning("Please enter both keys.")
 
    st.divider()
 
    valid_invs = [v for v in st.session_state.invoices if v.get("_status") == "valid"]
    if valid_invs:
        st.markdown("### Invoices")
        for i, inv in enumerate(valid_invs):
            label = inv.get("_filename", "?")[:26]
            if st.button(label, key=f"nav_{i}", use_container_width=True):
                st.session_state.current_idx = i
                st.rerun()
 
        st.divider()
        st.markdown("### Session Summary")
        total_val = sum(v.get("total") or 0 for v in valid_invs)
        vendors   = set(v.get("vendor", "?") for v in valid_invs)
        overdue   = sum(1 for v in valid_invs
                        if (v.get("_financial") or {}).get("urgency") in ("overdue", "critical"))
        dupes     = sum(1 for v in valid_invs
                        if (v.get("_duplicate") or {}).get("isDuplicate"))
 
        col1, col2 = st.columns(2)
        col1.metric("Invoices", len(valid_invs))
        col2.metric("Vendors",  len(vendors))
        st.metric("Total value", fmt_money(total_val))
        if overdue: st.error(f"{overdue} overdue invoice(s)")
        if dupes:   st.warning(f"{dupes} possible duplicate(s)")


tab_upload, tab_analysis, tab_analytics = st.tabs(["Upload", "Analysis", "Analytics"])


with tab_upload:
    st.markdown("## Upload Invoices")
 
    if not st.session_state.api_ready:
        st.info("Enter your API keys in the sidebar and click Connect first.")
    else:
        uploaded = st.file_uploader(
            "Drop your invoice files here (PDF, XLSX, DOCX, CSV, TXT, JSON)",
            accept_multiple_files=True,
            type=["pdf", "xlsx", "xls", "docx", "doc", "csv", "tsv", "txt", "json"],
        )
 
        if uploaded and st.button("Process invoices", type="primary"):
            progress = st.progress(0)
            status   = st.empty()
 
            for file_idx, uf in enumerate(uploaded):
                status.text(f"Parsing {uf.name}...")
                file_bytes = uf.read()
                chunks     = parse_file(uf.name, file_bytes)
 
                for chunk_idx, chunk_text in enumerate(chunks):
                    fname = uf.name if len(chunks) == 1 else f"{uf.name} [invoice {chunk_idx+1}]"
                    status.text(f"Classifying {fname}...")
 
                    inv = process_invoice(
                        st.session_state.groq_client,
                        chunk_text,
                        fname,
                        st.session_state.invoices,
                    )
 
                    # Grey zone confirmation
                    if inv.get("_status") == "review":
                        conf = inv.get("confidence", 0)
                        st.warning(f"{fname} — {conf}% confidence. Is this an invoice?")
                        cy, cn = st.columns(2)
                        if cy.button("Yes, it is an invoice", key=f"y_{file_idx}_{chunk_idx}"):
                            inv["_status"]   = "valid"
                            inv["isInvoice"] = True
                        if cn.button("Not an invoice", key=f"n_{file_idx}_{chunk_idx}"):
                            inv["_status"]   = "invalid"
 
                    # Index valid invoices into Pinecone
                    if inv.get("_status") == "valid":
                        iid = invoice_id(inv)
                        if iid not in st.session_state.indexed_ids:
                            status.text(f"Indexing {fname} into Pinecone...")
                            index_invoice(
                                st.session_state.pinecone_key,
                                iid,
                                chunk_text,
                                {"vendor": inv.get("vendor", ""), "filename": fname},
                            )
                            st.session_state.indexed_ids.add(iid)
                            st.session_state.chat_histories[iid] = []
 
                    st.session_state.invoices.append(inv)
                progress.progress((file_idx + 1) / len(uploaded))
 
            status.text("Done.")
            time.sleep(0.5)
            st.rerun()
 
        if st.session_state.invoices:
            st.markdown("---")
            st.markdown("### Processed files")
            for inv in st.session_state.invoices:
                status = inv.get("_status", "invalid")
                tag    = "[valid]" if status == "valid" else "[review]" if status == "review" else "[invalid]"
                vendor = inv.get("vendor") or "Unknown vendor"
                total  = fmt_money(inv.get("total"), inv.get("currency", "USD"))
                conf   = inv.get("confidence", 0)
                st.markdown(f"**{inv.get('_filename','')}** {tag} — {vendor} — {total} — {conf}% confidence")


with tab_analysis:
    valid_invs = [v for v in st.session_state.invoices if v.get("_status") == "valid"]
 
    if not valid_invs:
        st.info("Upload and process at least one valid invoice first.")
    else:
        idx = st.session_state.current_idx % len(valid_invs)
        inv = valid_invs[idx]
        iid = invoice_id(inv)
        cur = inv.get("currency", "USD")
        fin = inv.get("_financial") or {}
        val = inv.get("_validation") or {}
        dup = inv.get("_duplicate") or {}
 
        # preview
        vendor = inv.get("vendor") or "Unknown vendor"
        total  = fmt_money(inv.get("total"), cur)
        conf   = inv.get("confidence", 0)
 
        st.markdown(f"### {inv.get('_filename','')}")
        st.caption(
            f"This looks like an invoice from **{vendor}** for **{total}** "
            f"dated **{inv.get('date','N/A')}**, due **{inv.get('dueDate','N/A')}** — "
            f"Invoice #{inv.get('invoiceNumber','N/A')} — Confidence: {conf}%"
        )
 
        if dup.get("isDuplicate"):
            st.error(f"Possible duplicate — matches {dup.get('matchedWith','')}")
 
        # metrics
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total",    total)
        c2.metric("Subtotal", fmt_money(inv.get("subtotal"), cur))
        c3.metric("Tax",      fmt_money(inv.get("taxAmount"), cur))
 
        days_left = fin.get("daysUntilDue")
        urg       = urgency_label(fin.get("urgency", ""))
        c4.metric("Due", f"{urg} {fin.get('urgencyLabel','N/A')}" if days_left is not None else "N/A")
 
        back_tax = val.get("backCalcTaxRate")
        c5.metric("Tax rate", f"{back_tax}%" if back_tax else "N/A")
 
        epd = fin.get("earlyPayDiscount")
        if epd:
            st.success(f"Early pay offer: {epd.get('label','')}")
 
        if val.get("roundNumber"):
            st.warning("Round number total detected — worth verifying manually.")
 
        # checks
        flags = val.get("flags") or []
        if flags:
            st.markdown("**Validation checks**")
            html = ""
            for flag in flags:
                css = {"ok": "flag-ok", "error": "flag-err", "warn": "flag-warn"}.get(flag["type"], "flag-ok")
                html += f'<span class="{css}">{flag["msg"]}</span> '
            st.markdown(html, unsafe_allow_html=True)
 
        # line
        line_items = inv.get("lineItems") or []
        li_errors  = {e["line"] - 1 for e in (val.get("lineItemErrors") or [])}
 
        if line_items:
            st.markdown("---")
            st.markdown("**Line items**")
            rows = []
            for i, li in enumerate(line_items):
                qty      = li.get("qty") or 1
                price    = li.get("unitPrice") or 0
                listed   = li.get("lineTotal") or 0
                expected = round(qty * price, 2)
                rows.append({
                    "Description": li.get("description", ""),
                    "Qty":         qty,
                    "Unit price":  fmt_money(price, cur),
                    "Expected":    fmt_money(expected, cur),
                    "Listed":      fmt_money(listed, cur),
                    "Status":      "mismatch" if i in li_errors else "ok",
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)
 
            prices = [li.get("unitPrice") for li in line_items if li.get("unitPrice")]
            if prices:
                st.caption(
                    f"Avg unit price: {fmt_money(sum(prices)/len(prices), cur)} — "
                    f"Min: {fmt_money(min(prices), cur)} — "
                    f"Max: {fmt_money(max(prices), cur)}"
                )
 
        # chat
        st.markdown("---")
        st.markdown("### Ask anything about this invoice")
 
        history = st.session_state.chat_histories.get(iid, [])
 
        # d-sug
        if not history:
            defaults = suggest_questions(inv)
            st.markdown("**Suggested questions to get started:**")
            for s in defaults:
                st.markdown(f'<div class="sug-box">{s}</div>', unsafe_allow_html=True)
            st.markdown("")
 
        # s-sug
        for msg in history:
            role    = msg["role"]
            content = re.sub(r'Relevant invoice context:.*?Question: ', '', msg["content"], flags=re.DOTALL).strip()
            if role == "user":
                st.chat_message("user").write(content)
            else:
                st.chat_message("assistant").write(content)
 
        if history:
            last_q = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
            last_a = next((m["content"] for m in reversed(history) if m["role"] == "assistant"), "")
            try:
                raw = st.session_state.groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    temperature=0.5,
                    max_tokens=200,
                    messages=[{"role": "user", "content":
                        f"Based on this invoice Q&A, suggest 3 short follow-up questions.\n"
                        f"Last question: {last_q}\nLast answer: {last_a}\n"
                        f"Respond ONLY with a JSON array of 3 strings. Raw JSON only, no markdown."
                    }]
                ).choices[0].message.content.strip()
                raw         = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
                suggestions = json.loads(raw)
                st.markdown("**Follow-up suggestions:**")
                for s in suggestions:
                    st.markdown(f'<div class="sug-box">{s}</div>', unsafe_allow_html=True)
            except Exception:
                pass
 
        # input
        user_q = st.chat_input("Ask anything about this invoice...")
        if user_q:
            if not st.session_state.api_ready:
                st.error("API not connected.")
            else:
                with st.spinner("Thinking..."):
                    answer = answer_question(
                        st.session_state.groq_client,
                        st.session_state.pinecone_key,
                        iid,
                        inv,
                        user_q,
                        history,
                    )
                history.append({"role": "user",      "content": user_q})
                history.append({"role": "assistant",  "content": answer})
                st.session_state.chat_histories[iid] = history
                st.rerun()
 
        # optional export
        st.markdown("---")
        st.markdown("**Export**")
        e1, e2, e3 = st.columns(3)
 
        export_data = {k: v for k, v in inv.items() if not k.startswith("_raw")}
        e1.download_button(
            "Download JSON",
            data=json.dumps(export_data, indent=2, default=str),
            file_name=f"{inv.get('_filename','invoice')}.json",
            mime="application/json",
        )
 
        if line_items:
            import io, csv as csv_mod
            buf    = io.StringIO()
            writer = csv_mod.DictWriter(buf, fieldnames=["description", "qty", "unitPrice", "lineTotal"])
            writer.writeheader()
            writer.writerows(line_items)
            e2.download_button(
                "Download CSV",
                data=buf.getvalue(),
                file_name=f"{inv.get('_filename','invoice')}_lines.csv",
                mime="text/csv",
            )
 
        summary = f"""INVOICE SUMMARY
        
Vendor:        {inv.get('vendor','N/A')}
Invoice #:     {inv.get('invoiceNumber','N/A')}
Date:          {inv.get('date','N/A')}
Due date:      {inv.get('dueDate','N/A')}
Payment terms: {inv.get('paymentTerms','N/A')}
Currency:      {cur}
Subtotal:      {fmt_money(inv.get('subtotal'), cur)}
Tax:           {fmt_money(inv.get('taxAmount'), cur)}
Total:         {total}
 
Validation:
{chr(10).join('- ' + f['msg'] for f in flags) if flags else 'None'}
 
Financial:
{fin.get('urgencyLabel','N/A')}
{epd.get('label','No early pay discount') if epd else 'No early pay discount'}
"""
        e3.download_button(
            "Download Summary",
            data=summary,
            file_name=f"{inv.get('_filename','invoice')}_summary.txt",
            mime="text/plain",
        )


with tab_analytics:
    valid_invs = [v for v in st.session_state.invoices if v.get("_status") == "valid"]
 
    if len(valid_invs) < 1:
        st.info("Upload at least one valid invoice to see analytics.")
    else:
        st.markdown("### Running invoice log")
        running = 0
        rows    = []
        for inv in valid_invs:
            t       = inv.get("total") or 0
            running += t
            fin     = inv.get("_financial") or {}
            dup     = inv.get("_duplicate") or {}
            val     = inv.get("_validation") or {}
            flags   = []
            if val.get("roundNumber"):                                        flags.append("Round number")
            if dup.get("isDuplicate"):                                        flags.append("Duplicate")
            if fin.get("urgency") in ("overdue", "critical"):                flags.append("Overdue")
            rows.append({
                "File":          inv.get("_filename", ""),
                "Invoice #":     inv.get("invoiceNumber", "N/A"),
                "Vendor":        inv.get("vendor", "Unknown"),
                "Date":          inv.get("date", "N/A"),
                "Total":         fmt_money(t, inv.get("currency", "USD")),
                "Running total": fmt_money(running),
                "Flags":         ", ".join(flags) if flags else "--",
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)
 
        st.markdown("---")
        st.markdown("### Vendor spend breakdown")
        vendor_spend = {}
        for inv in valid_invs:
            v = inv.get("vendor") or "Unknown"
            vendor_spend[v] = vendor_spend.get(v, 0) + (inv.get("total") or 0)
 
        col_pie, col_bar = st.columns(2)
        fig_pie = px.pie(
            names=list(vendor_spend.keys()),
            values=list(vendor_spend.values()),
            title="Spend by vendor",
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        col_pie.plotly_chart(fig_pie, use_container_width=True)
 
        fig_bar = px.bar(
            x=list(vendor_spend.values()),
            y=list(vendor_spend.keys()),
            orientation="h",
            title="Vendor spend (bar)",
            labels={"x": "Total spend", "y": "Vendor"},
            color=list(vendor_spend.values()),
            color_continuous_scale="Blues",
        )
        fig_bar.update_layout(coloraxis_showscale=False)
        col_bar.plotly_chart(fig_bar, use_container_width=True)
 
        st.markdown("---")
        st.markdown("### Session KPIs")
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total invoices", len(valid_invs))
        k2.metric("Total value",    fmt_money(sum(v.get("total") or 0 for v in valid_invs)))
        k3.metric("Unique vendors", len(set(v.get("vendor", "?") for v in valid_invs)))
        k4.metric("Overdue",        sum(1 for v in valid_invs if (v.get("_financial") or {}).get("urgency") in ("overdue", "critical")))
        k5.metric("Duplicates",     sum(1 for v in valid_invs if (v.get("_duplicate") or {}).get("isDuplicate")))
 
        st.markdown("---")
        st.markdown("### Avg invoice value by vendor")
        vendor_counts = {}
        for inv in valid_invs:
            v = inv.get("vendor") or "Unknown"
            vendor_counts.setdefault(v, []).append(inv.get("total") or 0)
        avg_data = {v: sum(ts) / len(ts) for v, ts in vendor_counts.items()}
        fig_avg  = px.bar(
            x=list(avg_data.keys()),
            y=list(avg_data.values()),
            title="Average invoice value per vendor",
            labels={"x": "Vendor", "y": "Avg value"},
            color=list(avg_data.values()),
            color_continuous_scale="Teal",
        )
        fig_avg.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_avg, use_container_width=True)
 
