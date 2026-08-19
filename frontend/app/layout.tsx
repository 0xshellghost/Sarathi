import { NextPage } from 'next'
import Footer from '@/components/Footer'
import Header from '@/components/Header'


export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className="min-h-screen flex flex-col bg-gray-50">
        <Header />
        {/* main wrapper allows the content to expand and push the footer down */}
        <main className="flex-grow">
          {children}
        </main>
        <Footer />
      </body>
    </html>
  )
}
