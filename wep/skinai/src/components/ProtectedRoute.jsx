import { useState } from 'react';
import { useAuth } from '../context/AuthContext';
import AuthModal from './AuthModal';

export default function ProtectedRoute({ children }) {
  const { isLoggedIn } = useAuth();
  const [showModal, setShowModal] = useState(true);

  if (isLoggedIn) return children;

  return (
    <div style={{ paddingTop: 64, minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column', gap: 24, textAlign: 'center', padding: '100px 20px' }}>
      <div style={{ fontSize: '3rem' }}>🔐</div>
      <h2 style={{ fontSize: '1.4rem', fontWeight: 800 }}>로그인이 필요한 서비스입니다</h2>
      <p style={{ color: 'var(--text-light)', lineHeight: 1.7 }}>이 기능을 이용하려면 로그인 또는 회원가입이 필요해요.</p>
      <button
        onClick={() => setShowModal(true)}
        style={{ background: 'linear-gradient(135deg, var(--green-light), var(--green-dark))', color: '#fff', border: 'none', borderRadius: 50, padding: '12px 32px', fontSize: '0.95rem', fontWeight: 700, cursor: 'pointer', boxShadow: '0 4px 16px rgba(39,174,96,0.3)' }}
      >
        로그인하기
      </button>
      {showModal && <AuthModal onClose={() => setShowModal(false)} />}
    </div>
  );
}
