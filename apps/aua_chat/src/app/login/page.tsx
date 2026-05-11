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
    const res = await signIn('credentials', {
      username,
      password,
      redirect: false,
    })
    setLoading(false)
    if (res?.ok) {
      router.push('/')
      router.refresh()
    } else {
      setError('Invalid username or password.')
    }
  }

  return (
    <div className="min-h-screen bg-[#f7f6f2] flex items-center justify-center">
      <div className="bg-white rounded-2xl border border-[#e4e1da] shadow-sm p-10 w-full max-w-sm">
        {/* Logo / Title */}
        <div className="mb-8 text-center">
          <div className="inline-flex items-center gap-2 mb-3">
            <div className="w-8 h-8 rounded-lg bg-[#4338ca] flex items-center justify-center">
              <span className="text-white text-sm font-bold font-mono">A</span>
            </div>
            <span className="font-semibold text-[#18181b] text-lg">AUA Framework</span>
          </div>
          <p className="text-[#6b7280] text-sm">Sign in to your workspace</p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-[#18181b] mb-1">
              Username
            </label>
            <input
              type="text"
              value={username}
              onChange={e => setUsername(e.target.value)}
              placeholder="admin"
              autoComplete="username"
              required
              className="w-full px-3 py-2 rounded-lg border border-[#e4e1da] bg-white text-sm
                         focus:outline-none focus:ring-2 focus:ring-[#4338ca] focus:border-transparent
                         placeholder:text-[#9ca3af]"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-[#18181b] mb-1">
              Password
            </label>
            <input
              type="password"
              value={password}
              onChange={e => setPassword(e.target.value)}
              autoComplete="current-password"
              required
              className="w-full px-3 py-2 rounded-lg border border-[#e4e1da] bg-white text-sm
                         focus:outline-none focus:ring-2 focus:ring-[#4338ca] focus:border-transparent"
            />
          </div>

          {error && (
            <p className="text-sm text-red-600 bg-red-50 rounded-lg px-3 py-2">{error}</p>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full py-2.5 px-4 rounded-lg bg-[#4338ca] text-white text-sm font-medium
                       hover:bg-[#3730a3] disabled:opacity-60 disabled:cursor-not-allowed
                       transition-colors"
          >
            {loading ? 'Signing in…' : 'Sign in'}
          </button>
        </form>

        <p className="mt-6 text-center text-xs text-[#9ca3af]">
          Default: admin / aua-admin
        </p>
      </div>
    </div>
  )
}
