import Head from 'next/head'

export default function Layout({ children }) {
  return (
    <div className="min-h-screen bg-gray-100">
      <Head>
        <title>SDK Analyzer</title>
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <main className="container mx-auto px-4 py-8">
        {children}
      </main>
    </div>
  )
}