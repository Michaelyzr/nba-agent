"""Read-only providers. Each failure stays visible; absence is not recovery."""
import html
import io
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse
from xml.etree import ElementTree
from zoneinfo import ZoneInfo

import httpx

ESPN_INJURIES = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
ESPN_RSS = "https://www.espn.com/espn/rss/nba/news"
PDF_PATTERN = re.compile(r"^https://ak-static\.cms\.nba\.com/referee/injury/"
                         r"Injury-Report_(\d{4}-\d{2}-\d{2})_(\d{2})_(\d{2})(AM|PM)\.pdf$")
STATUSES = {"Out": "out", "Doubtful": "doubtful", "Questionable": "questionable",
            "Probable": "probable", "Available": "available", "Day-To-Day": "questionable"}


async def download(url, *, maximum=8_000_000):
    # Only fixed provider endpoints, official PDFs and official report directories.
    allowed = (url in {ESPN_INJURIES, ESPN_RSS} or bool(PDF_PATTERN.fullmatch(url))
               or bool(re.fullmatch(r"https://official\.nba\.com/nba-injury-report-\d{4}-\d{2}-season/", url)))
    if not allowed:
        raise ValueError("仅允许官方 NBA 伤病报告链接")
    async with httpx.AsyncClient(timeout=25, follow_redirects=False) as client:
        async with client.stream("GET", url, headers={"User-Agent": "NBA-Research-Agent/1.0"}) as response:
            response.raise_for_status()
            body = bytearray()
            async for chunk in response.aiter_bytes():
                body.extend(chunk)
                if len(body) > maximum:
                    raise ValueError("来源文件超过大小限制")
            return bytes(body)


def report_time(url):
    match = PDF_PATTERN.fullmatch(url)
    if not match:
        raise ValueError("官方 PDF 链接格式不正确")
    day, hour, minute, ampm = match.groups()
    local = datetime.strptime(f"{day} {hour}:{minute}{ampm}", "%Y-%m-%d %I:%M%p")
    return local.replace(tzinfo=ZoneInfo("America/New_York")).astimezone(timezone.utc)


def parse_pdf(content):
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(content))
    if len(reader.pages) > 60:
        raise ValueError("报告页数超过限制")
    output, columns = [], None
    for page in reader.pages:
        words = []
        def visitor(text, cm, tm, font, size):
            value = text.strip()
            if value:
                # NBA PDFs can flip the page's Y axis. pypdf layout mode then
                # reverses the table; transform coordinates before ordering.
                x = tm[4]*cm[0] + tm[5]*cm[2] + cm[4]
                y = tm[4]*cm[1] + tm[5]*cm[3] + cm[5]
                words.append((x, y, value))
        page.extract_text(visitor_text=visitor)
        anchors = {word: x for x, y, word in words if word in {"Player", "Current", "Reason"}}
        if len(anchors) == 3:
            columns = anchors
        if not columns:
            raise ValueError("官方 PDF 表头排版无法识别")
        status_rows = sorted({round(y, 1) for x, y, word in words
                              if columns["Current"]-2 <= x < columns["Reason"]-2
                              and word in STATUSES}, reverse=True)
        groups = []
        for x, y, word in sorted(words, key=lambda w: (-w[1], w[0])):
            if x >= columns["Reason"]-2:
                continue
            if not groups or abs(groups[-1][0]-y) > 2:
                groups.append((y, []))
            groups[-1][1].append((x, word))
        for y, cells in groups:
            cells.sort()
            line, last_x = "", None
            for x, word in cells:
                # Separate table columns while preserving spaces inside names.
                separator = "  " if last_x is not None and x-last_x > 100 else " "
                line += (separator if line else "") + word
                last_x = x
            row_y = next((v for v in status_rows if abs(v-y) <= 2), None)
            if row_y is not None:
                i = status_rows.index(row_y)
                upper = (status_rows[i-1]+row_y)/2 if i else row_y+16
                lower = (row_y+status_rows[i+1])/2 if i+1 < len(status_rows) else row_y-25
                reasons = sorted((w for w in words if w[0] >= columns["Reason"]-2 and lower < w[1] <= upper),
                                 key=lambda w: (-w[1], w[0]))
                line += "  " + " ".join(w[2] for w in reasons)
            output.append(line)
    return "\n".join(output)


def parse_report(text, teams):
    """Parse layout text, carrying explicit date/matchup/team across rows/pages.

    Missing/NOT YET SUBMITTED teams produce no player assertions. Unknown layouts
    fail closed instead of shifting columns and inventing a status.
    """
    date, matchup, team = None, None, None
    result, coverage = [], {}
    names = {re.sub(r"\s", "", t.full_name).lower(): t.full_name for t in teams}
    for line in text.splitlines():
        if "Injury Report:" in line or "Game Date" in line or "Page " in line:
            continue
        day_match = re.search(r"\b(\d{2}/\d{2}/\d{4})\b", line)
        if day_match:
            date = datetime.strptime(day_match[1], "%m/%d/%Y").date().isoformat()
        game_match = re.search(r"\b([A-Z]{2,3})@([A-Z]{2,3})\b", line)
        if game_match:
            matchup = list(game_match.groups())
            team = None
        compact = re.sub(r"\s", "", line).lower()
        for key, name in names.items():
            if key in compact:
                team = name
                break
        if date and matchup and team:
            scope = (date, tuple(matchup), team)
            if "notyetsubmitted" in compact:
                coverage[scope] = False
            elif "none" in compact:
                coverage.setdefault(scope, True)
        status_match = re.search(r"\b(Out|Doubtful|Questionable|Probable|Available)\b", line)
        if not status_match:
            continue
        if not (date and matchup and team):
            raise ValueError("官方报告出现缺少日期、对阵或球队的状态行")
        left = line[:status_match.start()]
        # Official player format Last, First, including suffixes and hyphens.
        player_match = re.search(r"([\wÀ-ÿ.'’\-]+(?:\s+[\wÀ-ÿ.'’\-]+)*,\s*"
                                 r"[\wÀ-ÿ.'’\-]+(?:\s+[\wÀ-ÿ.'’\-]+)*)\s*$", left)
        if not player_match:
            raise ValueError("官方报告出现无法解析球员姓名的状态行")
        player = player_match[1].strip()
        # Layout padding separates team and player columns; prevent team prefix.
        player = re.split(r"\s{2,}", player)[-1]
        reason = line[status_match.end():].strip()
        # G League assignment is a roster event, not a medical injury assertion.
        kind = "role_change" if "gleague" in re.sub(r"\s", "", reason).lower() else "injury"
        result.append({"player_name": player, "team_name": team, "game_date": date,
                       "matchup": matchup, "event_type": kind,
                       "player_status": STATUSES[status_match[1]], "minutes_limit": None,
                       "quote": line.strip(), "reason": reason})
        coverage.setdefault((date, tuple(matchup), team), True)
    for (day, pair, name), submitted in coverage.items():
        result.append({"player_name": None, "team_name": name, "game_date": day,
                       "matchup": list(pair), "event_type": "report_coverage", "player_status": "unknown",
                       "submitted": submitted, "minutes_limit": None,
                       "quote": f"{day} {pair[0]}@{pair[1]} {name}",
                       "reason": "官方报告已提交；列明的球员必须逐一匹配" if submitted else "NOT YET SUBMITTED · 未提交"})
    return result


def parse_espn_injuries(payload):
    rows = []
    if not isinstance(payload.get("injuries"), list):
        raise ValueError("ESPN 伤病响应格式变化")
    for team in payload["injuries"]:
        for injury in team.get("injuries", []):
            published = injury.get("date")
            player = injury.get("athlete", {}).get("displayName")
            status = STATUSES.get(injury.get("status"))
            if not (published and player and status):
                continue
            try:
                when = datetime.fromisoformat(published.replace("Z", "+00:00"))
                if when.tzinfo is None:
                    continue
            except ValueError:
                continue
            reason = injury.get("shortComment") or injury.get("longComment") or injury["status"]
            rows.append({"player_name": player, "team_name": team["displayName"],
                         "game_date": None, "matchup": None, "event_type": "injury",
                         "player_status": status, "minutes_limit": None,
                         "published_at": when.astimezone(timezone.utc).isoformat(),
                         "quote": reason, "reason": reason})
    return rows


def parse_rss(content):
    if b"<!DOCTYPE" in content.upper() or b"<!ENTITY" in content.upper():
        raise ValueError("不支持带实体定义的新闻源")
    root = ElementTree.fromstring(content)
    rows = []
    for item in root.findall("./channel/item")[:50]:
        url = item.findtext("link", "")
        if urlparse(url).hostname not in {"www.espn.com", "espn.com"}:
            continue
        try:
            when = parsedate_to_datetime(item.findtext("pubDate", ""))
            if when.tzinfo is None:
                continue
        except (ValueError, TypeError):
            continue
        title = html.unescape(item.findtext("title", ""))
        body = html.unescape(re.sub(r"<[^>]+>", " ", item.findtext("description", "")))
        rows.append({"url": url, "provider": "espn_rss", "title": title,
                     "body": f"{title}\n{body}", "published_at": when.astimezone(timezone.utc).isoformat()})
    return rows


class ReportLinks(HTMLParser):
    def __init__(self, base):
        super().__init__(); self.base = base; self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            url = urljoin(self.base, dict(attrs).get("href", ""))
            if PDF_PATTERN.fullmatch(url):
                self.links.append(url)


async def latest_official(season, now):
    url = f"https://official.nba.com/nba-injury-report-{season}-season/"
    parser = ReportLinks(url)
    parser.feed((await download(url)).decode("utf-8", errors="replace"))
    recent = [u for u in parser.links if 0 <= (now-report_time(u)).total_seconds() <= 48*3600]
    if not recent:
        raise ValueError("官方目录未发现最近 48 小时报告；不能确认当前伤病")
    return max(recent, key=report_time)
