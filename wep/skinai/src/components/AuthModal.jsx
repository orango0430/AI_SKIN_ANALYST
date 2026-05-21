import { useState } from 'react';
import { useAuth } from '../context/AuthContext';
import './AuthModal.css';

export default function AuthModal({ defaultTab = 'login', onClose }) {
  const [tab, setTab] = useState(defaultTab);
  const [alert, setAlert] = useState({ type: '', msg: '' });
  const { login, signup } = useAuth();

  // 로그인 상태
  const [loginEmail, setLoginEmail] = useState('');
  const [loginPw, setLoginPw] = useState('');
  const [loginErrors, setLoginErrors] = useState({});

  // 회원가입 상태
  const [signupName, setSignupName] = useState('');
  const [signupEmail, setSignupEmail] = useState('');
  const [signupPw, setSignupPw] = useState('');
  const [signupPw2, setSignupPw2] = useState('');
  const [agree, setAgree] = useState(false);
  const [signupErrors, setSignupErrors] = useState({});
  const [pwStrength, setPwStrength] = useState({ pct: 0, color: '#e0e0e0', label: '비밀번호를 입력하세요' });
  const [showPw, setShowPw] = useState(false);
  const [loading, setLoading] = useState(false);

  const validateEmail = (e) => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(e);

  const checkPwStrength = (pw) => {
    let score = 0;
    if (pw.length >= 8) score++;
    if (/[A-Z]/.test(pw)) score++;
    if (/[0-9]/.test(pw)) score++;
    if (/[^A-Za-z0-9]/.test(pw)) score++;
    const levels = [
      { pct: 0,   color: '#e0e0e0', label: '비밀번호를 입력하세요' },
      { pct: 25,  color: '#e74c3c', label: '너무 짧아요' },
      { pct: 50,  color: '#e67e22', label: '보통' },
      { pct: 75,  color: '#f1c40f', label: '좋아요' },
      { pct: 100, color: '#27ae60', label: '매우 안전합니다 ✓' },
    ];
    setPwStrength(pw.length === 0 ? levels[0] : levels[Math.max(1, score)]);
  };

  const switchTab = (t) => { setTab(t); setAlert({ type: '', msg: '' }); setLoginErrors({}); setSignupErrors({}); };

  const handleLogin = async (e) => {
    e.preventDefault();
    const errs = {};
    if (!validateEmail(loginEmail)) errs.email = '올바른 이메일을 입력해주세요.';
    if (!loginPw) errs.pw = '비밀번호를 입력해주세요.';
    if (Object.keys(errs).length) { setLoginErrors(errs); return; }
    setLoading(true); setAlert({ type: '', msg: '' });
    try {
      const user = await login(loginEmail, loginPw);
      setAlert({ type: 'success', msg: `${user.name}님, 환영합니다! 🎉` });
      setTimeout(onClose, 1200);
    } catch (err) {
      setAlert({ type: 'error', msg: err.message });
    } finally { setLoading(false); }
  };

  const handleSignup = async (e) => {
    e.preventDefault();
    const errs = {};
    if (signupName.length < 2) errs.name = '이름을 2자 이상 입력해주세요.';
    if (!validateEmail(signupEmail)) errs.email = '올바른 이메일을 입력해주세요.';
    if (signupPw.length < 8) errs.pw = '비밀번호를 8자 이상 입력해주세요.';
    if (signupPw !== signupPw2) errs.pw2 = '비밀번호가 일치하지 않습니다.';
    if (!agree) errs.agree = '약관에 동의해주세요.';
    if (Object.keys(errs).length) { setSignupErrors(errs); return; }
    setLoading(true); setAlert({ type: '', msg: '' });
    try {
      await signup(signupName, signupEmail, signupPw);
      setAlert({ type: 'success', msg: '회원가입 완료! SkinAI에 오신 걸 환영해요 🌿' });
      setTimeout(onClose, 1400);
    } catch (err) {
      setAlert({ type: 'error', msg: err.message });
    } finally { setLoading(false); }
  };

  return (
    <div className="modal-overlay" onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div className="auth-modal">
        <div className="modal-header">
          <button className="modal-close" onClick={onClose}>✕</button>
          <div className="modal-logo">🌿</div>
          <h2>SkinAI에 오신 걸 환영해요</h2>
          <p>AI 피부 분석과 전문 상담을 무료로 이용하세요</p>
        </div>

        <div className="auth-tabs">
          <button className={`auth-tab ${tab === 'login' ? 'active' : ''}`} onClick={() => switchTab('login')}>로그인</button>
          <button className={`auth-tab ${tab === 'signup' ? 'active' : ''}`} onClick={() => switchTab('signup')}>회원가입</button>
        </div>

        <div className="auth-form-wrap">
          {alert.msg && <div className={`auth-alert ${alert.type}`}><span>{alert.type === 'error' ? '⚠️' : '✅'}</span><span>{alert.msg}</span></div>}

          {/* 로그인 */}
          {tab === 'login' && (
            <form onSubmit={handleLogin}>
              <div className="form-group">
                <label>이메일</label>
                <div className="input-wrap">
                  <span className="input-icon">📧</span>
                  <input type="email" placeholder="example@email.com" value={loginEmail} onChange={e => setLoginEmail(e.target.value)} className={loginErrors.email ? 'error' : ''} />
                </div>
                {loginErrors.email && <p className="err-msg">{loginErrors.email}</p>}
              </div>
              <div className="form-group">
                <label>비밀번호</label>
                <div className="input-wrap">
                  <span className="input-icon">🔒</span>
                  <input type={showPw ? 'text' : 'password'} placeholder="비밀번호를 입력하세요" value={loginPw} onChange={e => setLoginPw(e.target.value)} className={loginErrors.pw ? 'error' : ''} />
                  <button type="button" className="pw-toggle" onClick={() => setShowPw(p => !p)}>{showPw ? '🙈' : '👁'}</button>
                </div>
                {loginErrors.pw && <p className="err-msg">{loginErrors.pw}</p>}
              </div>
              <div className="forgot-pw"><a href="#">비밀번호를 잊으셨나요?</a></div>
              <button type="submit" className="auth-submit" disabled={loading}>{loading ? '로그인 중...' : '로그인'}</button>
              <p className="switch-tab">계정이 없으신가요? <span onClick={() => switchTab('signup')}>회원가입</span></p>
            </form>
          )}

          {/* 회원가입 */}
          {tab === 'signup' && (
            <form onSubmit={handleSignup}>
              <div className="form-group">
                <label>이름 (닉네임)</label>
                <div className="input-wrap">
                  <span className="input-icon">👤</span>
                  <input type="text" placeholder="이름을 입력하세요" value={signupName} onChange={e => setSignupName(e.target.value)} className={signupErrors.name ? 'error' : ''} />
                </div>
                {signupErrors.name && <p className="err-msg">{signupErrors.name}</p>}
              </div>
              <div className="form-group">
                <label>이메일</label>
                <div className="input-wrap">
                  <span className="input-icon">📧</span>
                  <input type="email" placeholder="example@email.com" value={signupEmail} onChange={e => setSignupEmail(e.target.value)} className={signupErrors.email ? 'error' : ''} />
                </div>
                {signupErrors.email && <p className="err-msg">{signupErrors.email}</p>}
              </div>
              <div className="form-group">
                <label>비밀번호</label>
                <div className="input-wrap">
                  <span className="input-icon">🔒</span>
                  <input type={showPw ? 'text' : 'password'} placeholder="8자 이상 입력하세요" value={signupPw} onChange={e => { setSignupPw(e.target.value); checkPwStrength(e.target.value); }} className={signupErrors.pw ? 'error' : ''} />
                  <button type="button" className="pw-toggle" onClick={() => setShowPw(p => !p)}>{showPw ? '🙈' : '👁'}</button>
                </div>
                <div className="pw-strength">
                  <div className="pw-bar"><div className="pw-fill" style={{ width: `${pwStrength.pct}%`, background: pwStrength.color }} /></div>
                  <span className="pw-label" style={{ color: pwStrength.color }}>{pwStrength.label}</span>
                </div>
                {signupErrors.pw && <p className="err-msg">{signupErrors.pw}</p>}
              </div>
              <div className="form-group">
                <label>비밀번호 확인</label>
                <div className="input-wrap">
                  <span className="input-icon">🔒</span>
                  <input type="password" placeholder="비밀번호를 다시 입력하세요" value={signupPw2} onChange={e => setSignupPw2(e.target.value)} className={signupErrors.pw2 ? 'error' : ''} />
                </div>
                {signupErrors.pw2 && <p className="err-msg">{signupErrors.pw2}</p>}
              </div>
              <div className="form-check">
                <input type="checkbox" id="agree" checked={agree} onChange={e => setAgree(e.target.checked)} />
                <label htmlFor="agree"><a href="#">이용약관</a> 및 <a href="#">개인정보처리방침</a>에 동의합니다 (필수)</label>
              </div>
              {signupErrors.agree && <p className="err-msg" style={{ marginTop: '-10px', marginBottom: '12px' }}>{signupErrors.agree}</p>}
              <button type="submit" className="auth-submit" disabled={loading}>{loading ? '가입 중...' : '회원가입'}</button>
              <p className="switch-tab">이미 계정이 있으신가요? <span onClick={() => switchTab('login')}>로그인</span></p>
            </form>
          )}
        </div>
      </div>
    </div>
  );
}
