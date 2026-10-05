import assert from 'node:assert/strict';
import { streamAnalysis } from '../src/api.js';

const original = globalThis.fetch;
const encoder = new TextEncoder();
function fakeStream(text) {
  const bytes = encoder.encode(text);
  // Every byte arrives separately, splitting Chinese UTF-8 and SSE boundaries.
  return new Response(new ReadableStream({start(controller) {
    for(const byte of bytes) controller.enqueue(new Uint8Array([byte]));
    controller.close();
  }}),{headers:{'Content-Type':'text/event-stream'}});
}
try {
  globalThis.fetch=async()=>fakeStream('data: {"type":"meta","model":"test"}\n\ndata: {"type":"delta","text":"球队报告：中文"}\n\ndata: {"type":"done"}\n\n');
  const events=[];
  await streamAnalysis({}, {onEvent:event=>events.push(event)});
  assert.equal(events[1].text,'球队报告：中文');
  assert.deepEqual(events.map(e=>e.type),['meta','delta','done']);
  globalThis.fetch=async()=>fakeStream('data: {"type":"delta","text":"部分回答"}\n\n');
  await assert.rejects(streamAnalysis({}, {onEvent:()=>{}}),/连接中断/);
  globalThis.fetch=async()=>fakeStream('data: {"type":"error","message":"模型额度不足"}\n\n');
  await assert.rejects(streamAnalysis({}, {onEvent:()=>{}}),/模型额度不足/);
  globalThis.fetch=async()=>new Response(JSON.stringify({detail:'未配置密钥'}),{status:503});
  await assert.rejects(streamAnalysis({}, {onEvent:()=>{}}),/未配置密钥/);
  console.log('PASS: fragmented UTF-8/SSE, interrupted output, model failure, HTTP error');
} finally {globalThis.fetch=original;}
