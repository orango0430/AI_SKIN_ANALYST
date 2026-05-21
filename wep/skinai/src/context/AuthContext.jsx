import { createContext, useContext, useState, useEffect } from 'react';

const AuthContext = createContext(null);

// 배포 환경별로 Vercel에서 REACT_APP_API_BASE_URL 환경변수로 주입.
// 로컬 개발 fallback = localhost:8000.
const API_BASE_URL =
  process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

export function AuthProvider({ children }) {
  const [currentUser, setCurrentUser] = useState(null);

  // 앱 시작 시 localStorage에서 토큰 복원
  useEffect(() => {
    const saved = localStorage.getItem('skinai_user');
    if (saved) {
      try { setCurrentUser(JSON.parse(saved)); }
      catch { localStorage.removeItem('skinai_user'); }
    }
  }, []);

  // ── 로그인 ──────────────────────────────────
  const login = async (email, password) => {
    const res = await fetch(`${API_BASE_URL}/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || '로그인에 실패했습니다.');

    const user = { name: data.name, email: data.email, token: data.token };
    setCurrentUser(user);
    localStorage.setItem('skinai_user', JSON.stringify(user));
    return user;
  };

  // ── 회원가입 ─────────────────────────────────
  const signup = async (name, email, password) => {
    const res = await fetch(`${API_BASE_URL}/auth/signup`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, email, password })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || '회원가입에 실패했습니다.');

    const user = { name: data.name, email: data.email, token: data.token };
    setCurrentUser(user);
    localStorage.setItem('skinai_user', JSON.stringify(user));
    return user;
  };

  // ── 로그아웃 ─────────────────────────────────
  const logout = () => {
    setCurrentUser(null);
    localStorage.removeItem('skinai_user');
  };

  // ── API 요청 시 토큰 첨부 헬퍼 ──────────────
  const authFetch = async (url, options = {}) => {
    const headers = {
      'Authorization': `Bearer ${currentUser?.token}`,
      ...options.headers,
    };
    // FormData(이미지 업로드 등)가 아닐 때만 JSON Content-Type 설정
    if (!(options.body instanceof FormData)) {
      headers['Content-Type'] = headers['Content-Type'] || 'application/json';
    }
    const res = await fetch(`${API_BASE_URL}${url}`, {
      ...options,
      headers,
    });
    if (res.status === 401) {
      logout();
      throw new Error('로그인이 필요합니다.');
    }
    return res;
  };

  return (
    <AuthContext.Provider value={{
      currentUser,
      login,
      signup,
      logout,
      authFetch,
      isLoggedIn: !!currentUser
    }}>
      {children}
    </AuthContext.Provider>
  );
}

export const useAuth = () => useContext(AuthContext);
