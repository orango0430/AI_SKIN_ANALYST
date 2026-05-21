import { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import AuthModal from './AuthModal';
import './Navbar.css';

export default function Navbar() {
  const { currentUser, logout, isLoggedIn } = useAuth();
  const [showModal, setShowModal] = useState(false);
  const [modalTab, setModalTab] = useState('login');
  const [showDropdown, setShowDropdown] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();

  const openModal = (tab) => { setModalTab(tab); setShowModal(true); };

  const handleNav = (path, requireAuth = false) => {
    if (requireAuth && !isLoggedIn) { openModal('login'); return; }
    navigate(path);
  };

  const isActive = (path) => location.pathname === path;

  return (
    <>
      <nav className="navbar">
        <div className="nav-inner">
          <div className="nav-logo" onClick={() => navigate('/')}>
            🌿 <span>Skin<span className="logo-dot">AI</span></span>
          </div>
          <div className="nav-links">
            <span className={`nav-link ${isActive('/') ? 'active' : ''}`} onClick={() => navigate('/')}>홈</span>
            <span className={`nav-link ${isActive('/analysis') ? 'active' : ''}`} onClick={() => handleNav('/analysis', true)}>AI 피부 분석</span>
            <span className={`nav-link nav-cta ${isActive('/chat') ? 'active' : ''}`} onClick={() => handleNav('/chat', true)}>💬 AI 상담</span>
          </div>
          <div className="nav-auth">
            {isLoggedIn ? (
              <div className="nav-user-wrap">
                <div className="nav-user-info" onClick={() => setShowDropdown(p => !p)}>
                  <div className="nav-avatar">{currentUser.name.charAt(0).toUpperCase()}</div>
                  <span className="nav-username">{currentUser.name}</span>
                  <span className="nav-chevron">▾</span>
                  {showDropdown && (
                    <div className="user-dropdown">
                      <div className="dropdown-item" onClick={() => { handleNav('/analysis', true); setShowDropdown(false); }}>🔍 피부 분석</div>
                      <div className="dropdown-item" onClick={() => { handleNav('/chat', true); setShowDropdown(false); }}>💬 AI 상담</div>
                      <div className="dropdown-item" onClick={() => { handleNav('/mypage', true); setShowDropdown(false); }}>📁 마이페이지</div>
                      <div className="dropdown-divider" />
                      <div className="dropdown-item logout" onClick={() => { logout(); setShowDropdown(false); navigate('/'); }}>🚪 로그아웃</div>
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <div className="nav-guest">
                <button className="btn-login-nav" onClick={() => openModal('login')}>로그인</button>
                <button className="btn-signup-nav" onClick={() => openModal('signup')}>회원가입</button>
              </div>
            )}
          </div>
        </div>
      </nav>

      {showModal && <AuthModal defaultTab={modalTab} onClose={() => setShowModal(false)} />}
    </>
  );
}
