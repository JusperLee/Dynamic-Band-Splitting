"""Paper and author information shared by local and hosted studio interfaces."""
import html
import os
from urllib.parse import urlsplit

from ui_text import bi


PAPER_TITLE = "Learning Where to Split: Dynamic Band Splitting for Audio Source Separation"
CODE_URL = "https://github.com/JusperLee/Dynamic-Band-Splitting"
PROJECT_URL = "https://cslikai.cn/Dynamic-Band-Splitting/"


def research_panel():
    # A public paper URL can be added through the Space's environment settings.
    # Leave the entry unavailable until an actual paper URL has been provided.
    paper_url = os.environ.get("DBS_PAPER_URL", "").strip()
    if paper_url:
        parsed = urlsplit(paper_url)
        if parsed.scheme not in {"https", "http"} or not parsed.netloc:
            raise ValueError("DBS_PAPER_URL must be an absolute HTTP or HTTPS URL.")
        paper = link(paper_url, bi("Paper", "论文"))
    else:
        paper = f'<span class="research-link unavailable" aria-disabled="true">{bi("Paper · link coming soon", "论文 · 链接待补充")}</span>'

    return (
        '<section class="research-panel" aria-labelledby="paper-title">'
        f'<div class="eyebrow">{bi("THE RESEARCH BEHIND THE STUDIO", "声音工作室背后的研究")}</div>'
        f'<h2 id="paper-title">{PAPER_TITLE}</h2>'
        '<p class="research-authors">Bingzhe Zhang<sup>1*</sup> · '
        f'{link("https://cslikai.cn/", "Kai Li")}<sup>1,2*†</sup> · '
        f'{link("https://xlhu.cn/index.html", "Xiaolin Hu")}<sup>1,2,3†</sup></p>'
        f'<p class="research-affiliation"><sup>1</sup>{bi("Department of Computer Science and Technology, Institute for AI, BNRist, Tsinghua University", "清华大学计算机科学与技术系、人工智能研究院、北京信息科学与技术国家研究中心")}'
        f'<br><sup>2</sup>{bi("IDG/McGovern Institute for Brain Research, Tsinghua University", "清华大学 IDG/麦戈文脑科学研究院")}'
        f'<br><sup>3</sup>{bi("Chinese Institute for Brain Research (CIBR)", "北京脑科学与类脑研究所（CIBR）")}</p>'
        f'<p class="research-notes">{bi("* Equal contribution · † Corresponding authors", "* 共同第一作者 · † 通讯作者")}</p>'
        f'<nav class="research-links" aria-label="Research links" data-aria-en="Research links" data-aria-zh="研究资料链接">{paper}'
        f'{link(CODE_URL, bi("Code", "代码"))}'
        f'{link(PROJECT_URL, bi("Project page", "项目主页"))}'
        '</nav></section>'
    )


def link(url, label):
    return f'<a class="research-link" href="{html.escape(url, quote=True)}" target="_blank" rel="noopener noreferrer">{label}</a>'
