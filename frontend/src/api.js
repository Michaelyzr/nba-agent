export async function request(path, { method = 'GET', body, signal } = {}) {
  const response = await fetch(`/api/v1${path}`, {
    method, signal, headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  let payload;
  try { payload = await response.json(); }
  catch { throw new Error('后端没有返回有效数据，请确认 API 服务已启动。'); }
  if (!response.ok) {
    const detail = payload.detail;
    throw new Error(typeof detail === 'string' ? detail :
      Array.isArray(detail) ? detail.map(item => item.msg).join('；') :
      detail?.message || payload.message || `请求失败 (${response.status})`);
  }
  return payload.data ?? payload;
}

export const percent = value => value == null ? '—' : `${(value * 100).toFixed(1)}%`;
export const number = value => value == null ? '—' : Number(value).toFixed(2);
export const date = value => value ? new Intl.DateTimeFormat('zh-CN', {
  timeZone: 'Asia/Shanghai', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', hour12: false,
}).format(new Date(value)) : '—';

// Decode complete SSE frames, including multibyte Chinese text split across chunks.
export async function streamAnalysis(body, { signal, onEvent }) {
  const response = await fetch('/api/v1/analysis/chat', {
    method: 'POST', signal, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    const detail = payload.detail;
    const error = new Error(typeof detail === 'string' ? detail : Array.isArray(detail)
      ? detail.map(item => item.msg).join('；') : detail?.message || `请求失败 (${response.status})`);
    error.status = response.status;
    throw error;
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '', completed = false;
  try {
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });
      let boundary;
      while ((boundary = buffer.indexOf('\n\n')) !== -1) {
        const frame = buffer.slice(0, boundary); buffer = buffer.slice(boundary + 2);
        const data = frame.split('\n').filter(line => line.startsWith('data:')).map(line => line.slice(5).trim()).join('\n');
        if (!data) continue;
        const event = JSON.parse(data);
        if (event.type === 'error') throw new Error(event.message);
        if (event.type === 'done') completed = true;
        onEvent(event);
      }
      if (done) break;
    }
    if (!completed) throw new Error('回答连接中断；已显示的内容为部分结果，请重试。');
  } finally { await reader.cancel().catch(() => {}); reader.releaseLock(); }
}
