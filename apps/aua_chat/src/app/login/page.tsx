'use client'

import { signIn } from 'next-auth/react'
import { useState, FormEvent } from 'react'
import { useRouter } from 'next/navigation'

export default function LoginPage() {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const router = useRouter()

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    setLoading(true)
    setError('')
    const res = await signIn('credentials', { username, password, redirect: false })
    setLoading(false)
    if (res?.ok) { router.push('/'); router.refresh() }
    else setError('Invalid username or password.')
  }

  return (
    <div style={{
      minHeight: '100vh',
      background: 'var(--bg)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '1.5rem',
    }}>
      {/* Card — matches the HTML page's .page border style */}
      <div style={{
        background: 'var(--paper)',
        border: '1px solid var(--line)',
        borderRadius: '12px',
        padding: '2.5rem 2.25rem',
        width: '100%',
        maxWidth: '360px',
        boxShadow: '0 1px 3px rgba(0,0,0,.06)',
      }}>

        {/* Hero-style header */}
        <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
          {/* Badge — matches .hero-tag */}
          <span style={{
            display: 'inline-block',
            fontFamily: '"JetBrains Mono", monospace',
            fontSize: '.68rem',
            fontWeight: 600,
            letterSpacing: '.1em',
            textTransform: 'uppercase',
            background: 'var(--tag-bg)',
            color: 'var(--accent)',
            padding: '.2rem .65rem',
            borderRadius: '999px',
            marginBottom: '1rem',
          }}>
            AUA v1.0
          </span>

          <h1 style={{
            fontFamily: '"DM Serif Display", Georgia, serif',
            fontSize: '1.75rem',
            lineHeight: 1.2,
            margin: '0 0 .4rem',
            color: 'var(--ink)',
          }}>
            Framework Chat
          </h1>
          <p style={{ fontSize: '.88rem', color: 'var(--muted)', margin: 0 }}>
            Sign in to your workspace
          </p>
        </div>

        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          {(['Username', 'Password'] as const).map(field => (
            <div key={field}>
              <label style={{
                display: 'block',
                fontSize: '.82rem',
                fontWeight: 500,
                color: 'var(--ink)',
                marginBottom: '.35rem',
              }}>
                {field}
              </label>
              <input
                type={field === 'Password' ? 'password' : 'text'}
                value={field === 'Username' ? username : password}
                onChange={e => field === 'Username' ? setUsername(e.target.value) : setPassword(e.target.value)}
                placeholder={field === 'Username' ? 'admin' : ''}
                autoComplete={field === 'Username' ? 'username' : 'current-password'}
                required
                style={{
                  width: '100%',
                  padding: '.55rem .85rem',
                  borderRadius: '8px',
                  border: '1px solid var(--line)',
                  background: 'var(--paper)',
                  color: 'var(--ink)',
                  fontSize: '.9rem',
                  fontFamily: '"DM Sans", system-ui, sans-serif',
                  outline: 'none',
                  transition: 'border-color .15s, box-shadow .15s',
                }}
                onFocus={e => {
                  e.currentTarget.style.borderColor = 'var(--accent)'
                  e.currentTarget.style.boxShadow = '0 0 0 2px #c7d2fe'
                }}
                onBlur={e => {
                  e.currentTarget.style.borderColor = 'var(--line)'
                  e.currentTarget.style.boxShadow = 'none'
                }}
              />
            </div>
          ))}

          {error && (
            <div style={{
              padding: '.6rem .85rem',
              background: '#fff5f5',
              border: '1px solid #fecaca',
              borderLeft: '3px solid var(--accent3)',
              borderRadius: '0 6px 6px 0',
              fontSize: '.82rem',
              color: '#b91c1c',
            }}>
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={loading}
            style={{
              width: '100%',
              padding: '.65rem 1rem',
              marginTop: '.25rem',
              background: loading ? '#9ca3af' : 'var(--accent)',
              color: '#fff',
              border: 'none',
              borderRadius: '8px',
              fontSize: '.9rem',
              fontWeight: 500,
              fontFamily: '"DM Sans", system-ui, sans-serif',
              cursor: loading ? 'not-allowed' : 'pointer',
              transition: 'background .15s',
            }}
            onMouseEnter={e => { if (!loading) e.currentTarget.style.background = '#3730a3' }}
            onMouseLeave={e => { if (!loading) e.currentTarget.style.background = 'var(--accent)' }}
          >
            {loading ? 'Signing in…' : 'Sign in'}
          </button>
        </form>

        {/* Footer note — caption style from HTML pages */}
        <p style={{
          marginTop: '1.5rem',
          textAlign: 'center',
          fontSize: '.75rem',
          color: 'var(--muted)',
          fontFamily: '"JetBrains Mono", monospace',
        }}>
          Default: admin / aua-admin
        </p>

        {/* Link back to docs */}
        <p style={{ marginTop: '.5rem', textAlign: 'center', fontSize: '.78rem', color: 'var(--muted)' }}>
          <a href="whitepaper_v1_tmp.html" style={{ color: 'var(--accent)', textDecoration: 'none' }}>
            AUA Framework docs ↗
          </a>
        </p>
      </div>
    </div>
  )
}
