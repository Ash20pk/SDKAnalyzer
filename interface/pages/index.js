import { useState } from 'react'
import Layout from '../components/Layout'
import axios from 'axios'

export default function Home() {
  const [packageName, setPackageName] = useState('')
  const [version, setVersion] = useState('latest')
  const [analysis, setAnalysis] = useState(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState([])
  const [isSearching, setIsSearching] = useState(false)
  const [error, setError] = useState(null)

  const handleAnalyze = async () => {
    setIsAnalyzing(true)
    setError(null)
    try {
      const response = await axios.post('/api/analyze', { package_name: packageName, version })
      setAnalysis(response.data.result)
    } catch (error) {
      console.error('Error analyzing SDK:', error)
      setError(error.response?.data?.detail || 'An unexpected error occurred')
    }
    setIsAnalyzing(false)
  }

  const handleSearch = async () => {
    setIsSearching(true)
    setError(null)
    try {
      const response = await axios.post('/api/search', { package_name: packageName, version, query: searchQuery })
      setSearchResults(response.data.results)
    } catch (error) {
      console.error('Error searching SDK:', error)
      setError(error.response?.data?.detail || 'An unexpected error occurred')
    }
    setIsSearching(false)
  }

  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold mb-4">SDK Analyzer</h1>
      <div className="mb-4">
        <input
          type="text"
          value={packageName}
          onChange={(e) => setPackageName(e.target.value)}
          placeholder="Enter package name"
          className="w-full text-black p-2 border rounded"
        />
      </div>
      <div className="mb-4">
        <input
          type="text"
          value={version}
          onChange={(e) => setVersion(e.target.value)}
          placeholder="Enter version (or leave for latest)"
          className="w-full text-black p-2 border rounded"
        />
      </div>
      <button
        className="bg-blue-500 text-white px-4 py-2 rounded"
        onClick={handleAnalyze}
        disabled={!packageName || isAnalyzing}
      >
        {isAnalyzing ? 'Analyzing...' : 'Analyze SDK'}
      </button>

      {error && (
        <div className="mt-4 p-4 bg-red-100 border border-red-400 text-red-700 rounded">
          Error: {error}
        </div>
      )}

      {analysis && (
        <div className="mt-8">
          <h2 className="text-2xl font-bold mb-4">Analysis Result</h2>
          
          <Section title="AI Summary">
            <p>{analysis.ai_summary || 'No summary available.'}</p>
          </Section>

          <Section title="Potential Use Cases">
            {analysis.potential_use_cases && analysis.potential_use_cases.length > 0 ? (
              <ul className="list-disc pl-5">
                {analysis.potential_use_cases.map((useCase, index) => (
                  <li key={index}>{useCase}</li>
                ))}
              </ul>
            ) : (
              <p>No potential use cases identified.</p>
            )}
          </Section>

          <Section title="Package Details">
            <pre className="bg-gray-100 text-black p-4 rounded overflow-x-auto">
              {JSON.stringify(analysis, null, 2)}
            </pre>
          </Section>

          <Section title="Functions">
            {analysis.functions && analysis.functions.length > 0 ? (
              <ul className="list-disc pl-5">
                {analysis.functions.map((func, index) => (
                  <li key={index}>
                    <strong>{func.name}</strong>
                    <span className="text-white"> ({func.params.join(', ')})</span>
                    {func.docstring && <p className="text-sm text-white">{func.docstring}</p>}
                  </li>
                ))}
              </ul>
            ) : (
              <p>No functions found.</p>
            )}
          </Section>

          <Section title="Classes">
            {analysis.classes && analysis.classes.length > 0 ? (
              <ul className="list-disc pl-5">
                {analysis.classes.map((cls, index) => (
                  <li key={index}>
                    <strong>{cls.name}</strong>
                    {cls.docstring && <p className="text-sm text-white">{cls.docstring}</p>}
                    {cls.methods && cls.methods.length > 0 && (
                      <ul className="list-circle pl-5 mt-2">
                        {cls.methods.map((method, methodIndex) => (
                          <li key={methodIndex}>
                            <strong>{method.name}</strong>
                            <span className="text-white"> ({method.params.join(', ')})</span>
                            {method.docstring && <p className="text-sm text-white">{method.docstring}</p>}
                          </li>
                        ))}
                      </ul>
                    )}
                  </li>
                ))}
              </ul>
            ) : (
              <p>No classes found.</p>
            )}
          </Section>

          <Section title="Dependencies">
            <pre className="bg-gray-100 text-black p-4 rounded overflow-x-auto">
              {JSON.stringify(analysis.dependencies, null, 2)}
            </pre>
          </Section>
        </div>
      )}
    </div>
  )
}

function Section({ title, children }) {
  return (
    <div className="mb-6">
      <h3 className="text-xl font-semibold mb-2">{title}</h3>
      {children}
    </div>
  )
}