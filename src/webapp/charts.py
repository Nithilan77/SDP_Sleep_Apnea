"""Tiny dependency-free inline-SVG charts (server-side), themed via CSS variables in static/style.css."""
from __future__ import annotations
from html import escape
from markupsafe import Markup


def hbars(items, xmax=1.0, xmin=0.0, ticks=(0, .25, .5, .75, 1.0), width=720, row=34, fmt="{:.3f}", label_w=210):
    """items: list of dict(label, value|None, err=None (sd/halfwidth), cls='a', note='' ). value None -> 'pending' row."""
    h = row * len(items) + 34; pw = width - label_w - 70
    X = lambda v: label_w + (v - xmin) / (xmax - xmin) * pw
    s = [f'<svg viewBox="0 0 {width} {h}" class="chart" role="img" aria-label="bar chart">']
    for t in ticks:
        s.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="0" y2="{h-24}" class="grid"/><text x="{X(t):.1f}" y="{h-8}" class="tick" text-anchor="middle">{t:g}</text>')
    for i, it in enumerate(items):
        y = i * row + 6; v = it.get("value"); cls = it.get("cls", "a")
        s.append(f'<text x="{label_w-10}" y="{y+row/2-1:.1f}" class="lbl" text-anchor="end">{escape(it["label"])}</text>')
        if v is None:
            s.append(f'<text x="{label_w+6}" y="{y+row/2-1:.1f}" class="pending">pending - not yet run</text>'); continue
        s.append(f'<rect x="{X(xmin):.1f}" y="{y}" width="{max(X(v)-X(xmin),0):.1f}" height="{row-14}" rx="3" class="bar {cls}"/>')
        e = it.get("err")
        if e:
            s.append(f'<line x1="{X(v-e):.1f}" x2="{X(v+e):.1f}" y1="{y+(row-14)/2:.1f}" y2="{y+(row-14)/2:.1f}" class="whisk"/>'
                     f'<line x1="{X(v-e):.1f}" x2="{X(v-e):.1f}" y1="{y+3}" y2="{y+row-17}" class="whisk"/><line x1="{X(v+e):.1f}" x2="{X(v+e):.1f}" y1="{y+3}" y2="{y+row-17}" class="whisk"/>')
        s.append(f'<text x="{X(max(v+(e or 0), xmin))+6:.1f}" y="{y+row/2-1:.1f}" class="val">{fmt.format(v)}{" ±"+fmt.format(e) if e else ""}</text>')
    s.append("</svg>"); return Markup("".join(s))


def confusion(tp, fn, fp, tn, title=""):
    tot = tp + fn + fp + tn; c = lambda v: f"{v:,.0f}"
    return Markup(f'<table class="cm"><caption>{escape(title)}</caption><tr><th></th><th>pred. normal</th><th>pred. apnea</th></tr>'
                  f'<tr><th>true normal</th><td>{c(tn)}</td><td class="bad">{c(fp)}</td></tr><tr><th>true apnea</th><td class="bad">{c(fn)}</td><td>{c(tp)}</td></tr></table>')
