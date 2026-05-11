/**
 * AUA Chat UI — Authentication
 *
 * Uses NextAuth with CredentialsProvider for username/password login.
 *
 * Users are stored in AUA_USERS env var as JSON:
 *   AUA_USERS='[{"username":"admin","password":"aua-admin"}]'
 *
 * Or use the defaults for local dev:
 *   username: admin
 *   password: aua-admin
 *
 * In production: set AUA_USERS with strong passwords, or configure
 * an OAuth provider (Google, GitHub, etc.) via NextAuth.
 */

import NextAuth from 'next-auth'
import CredentialsProvider from 'next-auth/providers/credentials'

const DEFAULT_USERS = [
  { id: '1', username: 'admin', password: 'aua-admin', name: 'Admin' },
]

function getUsers() {
  try {
    const raw = process.env.AUA_USERS
    if (raw) return JSON.parse(raw)
  } catch {}
  return DEFAULT_USERS
}

const handler = NextAuth({
  providers: [
    CredentialsProvider({
      name: 'AUA Framework',
      credentials: {
        username: { label: 'Username', type: 'text', placeholder: 'admin' },
        password: { label: 'Password', type: 'password' },
      },
      async authorize(credentials) {
        if (!credentials?.username || !credentials?.password) return null
        const users = getUsers()
        const user = users.find(
          (u: any) =>
            u.username === credentials.username &&
            u.password === credentials.password
        )
        if (!user) return null
        return { id: user.id || '1', name: user.name || user.username, email: user.username }
      },
    }),
  ],
  pages: {
    signIn: '/login',
  },
  session: { strategy: 'jwt' },
  callbacks: {
    async jwt({ token, user }) {
      if (user) token.user = user
      return token
    },
    async session({ session, token }) {
      session.user = token.user as any
      return session
    },
  },
})

export { handler as GET, handler as POST }
