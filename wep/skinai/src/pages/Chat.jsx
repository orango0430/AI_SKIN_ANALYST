import { useState, useRef, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import './Chat.css';

function nowTime() {
  const d = new Date();
  return `${String(d.getHours()).padStart(2,'0')}:${String(d.getMinutes()).padStart(2,'0')}`;
}

const WELCOME = {
  role: 'bot',
  text: '안녕하세요! 저는 SkinAI 피부 상담사예요 😊\n\n피부 고민, 스킨케어 루틴, 성분 추천까지 무엇이든 편하게 물어보세요!',
  time: nowTime(),
  suggestions: ['피부 타입 파악', '여드름 관리', '건성 피부 성분'],
};

export default function Chat() {
  const [messages, setMessages] = useState([WELCOME]);
  const [input, setInput] = useState('');
  const [typing, setTyping] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [showHistory, setShowHistory] = useState(false);
  const chatRef = useRef();

  const { authFetch } = useAuth();

  // 과거 채팅 내역 불러오기
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await authFetch('/chat/history');
        if (!res.ok) throw new Error('내역을 불러오지 못했습니다.');
        const data = await res.json();
        if (cancelled) return;
        if (Array.isArray(data) && data.length > 0) {
          const restored = data.map((h) => ({
            role: h.role === 'user' ? 'user' : 'bot',
            text: h.message,
            time: '',
          }));
          setMessages([WELCOME, ...restored]);
        }
      } catch (e) {
        // 내역이 없거나 실패해도 환영 메시지는 유지
      } finally {
        if (!cancelled) setHistoryLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, [authFetch]);

  useEffect(() => {
    if (chatRef.current) chatRef.current.scrollTop = chatRef.current.scrollHeight;
  }, [messages, typing]);

  const send = async (text) => {
    const msg = text || input.trim();
    if (!msg || typing) return;
    setMessages(prev => [...prev, { role: 'user', text: msg, time: nowTime() }]);
    setInput('');
    setTyping(true);
    try {
      const res = await authFetch('/chat/message', {
        method: 'POST',
        body: JSON.stringify({ message: msg }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || '응답에 실패했습니다.');
      setMessages(prev => [...prev, { role: 'bot', text: data.message, time: nowTime() }]);
    } catch (err) {
      setMessages(prev => [...prev, { role: 'bot', text: `오류가 발생했어요: ${err.message}`, time: nowTime() }]);
    } finally {
      setTyping(false);
    }
  };

  const clearHistory = async () => {
    if (!window.confirm('대화 내역을 모두 삭제할까요?')) return;
    try {
      await authFetch('/chat/history', { method: 'DELETE' });
      setMessages([WELCOME]);
    } catch (e) {
      alert('삭제 실패: ' + e.message);
    }
  };

  const userMessages = messages.filter(m => m.role === 'user');

  return (
    <div className="chat-page">
      <div className="chat-layout">
        <div className="chat-header">
          <div className="chat-avatar">🤖</div>
          <div className="chat-header-info">
            <h2>SkinAI 피부 상담사</h2>
            <p>피부 고민이라면 무엇이든 물어보세요</p>
          </div>
          <div className="chat-status"><div className="status-dot" />온라인</div>
          <button
            className="chat-action-btn"
            title="과거 대화"
            onClick={() => setShowHistory(p => !p)}
          >
            📜
          </button>
          <button
            className="chat-action-btn"
            title="대화 초기화"
            onClick={clearHistory}
          >
            🗑️
          </button>
        </div>

        {showHistory && (
          <div className="history-panel">
            <div className="history-panel-title">📜 내가 보낸 질문 ({userMessages.length})</div>
            {userMessages.length === 0 ? (
              <div className="history-empty">아직 보낸 질문이 없어요.</div>
            ) : (
              <ul className="history-list">
                {userMessages.slice().reverse().map((m, i) => (
                  <li key={i}>{m.text}</li>
                ))}
              </ul>
            )}
          </div>
        )}

        <div className="chat-window" ref={chatRef}>
          {historyLoading && (
            <div className="history-loading">📥 이전 대화를 불러오는 중...</div>
          )}
          {messages.map((msg, i) => (
            <div key={i} className={`msg ${msg.role}`}>
              <div className="msg-avatar">{msg.role === 'bot' ? '🌿' : '👤'}</div>
              <div>
                <div className="msg-bubble">{msg.text.split('\n').map((line, j) => <span key={j}>{line}<br/></span>)}</div>
                {msg.suggestions && (
                  <div className="suggestions">
                    {msg.suggestions.map(s => <button key={s} className="suggestion-btn" onClick={() => send(s)}>{s}</button>)}
                  </div>
                )}
                {msg.time && <div className="msg-time">{msg.time}</div>}
              </div>
            </div>
          ))}
          {typing && (
            <div className="msg bot">
              <div className="msg-avatar">🌿</div>
              <div><div className="msg-bubble typing"><span/><span/><span/></div></div>
            </div>
          )}
        </div>

        <div className="chat-input-area">
          <div className="quick-btns">
            {['💧 수분크림 추천', '🔬 모공 관리', '☀️ 선크림 선택', '✨ 스킨케어 순서'].map(q => (
              <button key={q} className="quick-btn" onClick={() => send(q.replace(/^[^\s]+ /, ''))}>{q}</button>
            ))}
          </div>
          <div className="input-row">
            <textarea
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); } }}
              placeholder="피부 고민을 입력하세요..."
              rows={1}
              style={{ height: 'auto' }}
              onInput={e => { e.target.style.height = 'auto'; e.target.style.height = Math.min(e.target.scrollHeight, 120) + 'px'; }}
            />
            <button className="send-btn" onClick={() => send()}>➤</button>
          </div>
        </div>
      </div>
    </div>
  );
}
