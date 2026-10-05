"""Expose provider citations; never turn model-written URLs into evidence."""
from datetime import datetime, timezone
from urllib.parse import urlsplit


def value(item, name, default=None):
    return item.get(name, default) if isinstance(item, dict) else getattr(item, name, default)


def safe_url(url):
    if not isinstance(url, str) or len(url) > 4096 or any(c.isspace() or ord(c) < 32 for c in url):
        return None
    try:
        parsed = urlsplit(url)
        if parsed.scheme not in {'http', 'https'} or not parsed.hostname or parsed.username or parsed.password:
            return None
        return url
    except ValueError:
        return None


def web_evidence(response, *, completed_calls=0):
    sources, consulted, texts = [], [], []
    by_url = {}
    calls = 0
    for item in value(response, 'output', []) or []:
        if value(item, 'type') == 'web_search_call':
            if value(item, 'status') == 'completed':
                calls += 1
            action = value(item, 'action')
            for source in value(action, 'sources', []) or []:
                url = safe_url(value(source, 'url'))
                if url and url not in consulted:
                    consulted.append(url)
        if value(item, 'type') != 'message':
            continue
        for part in value(item, 'content', []) or []:
            if value(part, 'type') != 'output_text':
                continue
            text = value(part, 'text', '')
            replacements = {}
            for annotation in value(part, 'annotations', []) or []:
                if value(annotation, 'type') != 'url_citation':
                    continue
                url = safe_url(value(annotation, 'url'))
                if not url:
                    continue
                source = by_url.get(url)
                if source is None:
                    source = dict(id=f'W{len(sources)+1}', url=url,
                        title=value(annotation, 'title') or url, kind='web')
                    by_url[url] = source
                    sources.append(source)
                start, end = value(annotation, 'start_index'), value(annotation, 'end_index')
                if (isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(text)):
                    replacements.setdefault((start, end), []).append(source['id'])
            # Offsets are per text part. Normalize citation markers before joining
            # multiple messages/parts so frontend and exported reports agree.
            cursor, chunks = 0, []
            for (start, end), ids in sorted(replacements.items()):
                if start < cursor:
                    continue
                chunks.extend([text[cursor:start], ''.join(f'[{id}]' for id in dict.fromkeys(ids))])
                cursor = end
            chunks.append(text[cursor:])
            texts.append(''.join(chunks))
    return dict(type='web_sources', sources=sources, consulted_urls=consulted,
        text=''.join(texts), calls=max(calls, completed_calls),
        searched_at=datetime.now(timezone.utc).isoformat(),
        status='completed' if sources else 'no_sources')
