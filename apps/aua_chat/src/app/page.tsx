import { getServerSession } from 'next-auth'
import { redirect } from 'next/navigation'
import { ChatLayout } from '@/components/chat/ChatLayout'

export default async function Home() {
  const session = await getServerSession()
  if (!session) redirect('/login')
  return <ChatLayout />
}
