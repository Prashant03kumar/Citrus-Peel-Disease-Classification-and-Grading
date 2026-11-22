import { useEffect, useRef, useState } from 'react'
import { predictImage } from '../api'
import ResultCard from './ResultCard'

const MAX_FILE_SIZE = 10 * 1024 * 1024

function Uploader({ classes, onPrediction }) {
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState('')
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [dragging, setDragging] = useState(false)
  const fileInput = useRef(null)
  const previewRef = useRef('')
  const requestId = useRef(0)

  function clearPreview() {
    if (previewRef.current) URL.revokeObjectURL(previewRef.current)
    previewRef.current = ''
    setPreviewUrl('')
    setSelectedFile(null)
  }

  function chooseFile(file) {
    requestId.current += 1
    setLoading(false)
    setError('')
    setResult(null)
    clearPreview()
    if (!file) return
    if (!file.type.startsWith('image/')) {
      setError('That file is not an image. Choose a JPG, PNG, or another image file.')
      return
    }
    if (file.size > MAX_FILE_SIZE) {
      setError('This photo is larger than 10 MB. Choose a smaller image.')
      return
    }
    const nextPreview = URL.createObjectURL(file)
    previewRef.current = nextPreview
    setPreviewUrl(nextPreview)
    setSelectedFile(file)
  }

  useEffect(
    () => () => {
      if (previewRef.current) URL.revokeObjectURL(previewRef.current)
    },
    [],
  )

  async function classify(event) {
    event.preventDefault()
    if (!selectedFile || loading) return
    const thisRequest = ++requestId.current
    const fileToClassify = selectedFile
    setLoading(true)
    setError('')
    setResult(null)
    try {
      const prediction = await predictImage(fileToClassify)
      if (requestId.current !== thisRequest) return
      setResult(prediction)
      onPrediction(prediction, fileToClassify)
    } catch (predictionError) {
      if (requestId.current === thisRequest) {
        setError(predictionError.message || 'The photo could not be classified. Please try again.')
      }
    } finally {
      if (requestId.current === thisRequest) setLoading(false)
    }
  }

  function handleDrop(event) {
    event.preventDefault()
    setDragging(false)
    chooseFile(event.dataTransfer.files?.[0])
  }

  function handleZoneKeyDown(event) {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault()
      fileInput.current?.click()
    }
  }

  return (
    <>
      <div className="upload-layout">
        <form className="upload-panel panel" onSubmit={classify}>
          <div
            className={`drop-zone${dragging ? ' drop-zone-active' : ''}`}
            onDragEnter={(event) => {
              event.preventDefault()
              setDragging(true)
            }}
            onDragOver={(event) => event.preventDefault()}
            onDragLeave={(event) => {
              if (!event.currentTarget.contains(event.relatedTarget)) setDragging(false)
            }}
            onDrop={handleDrop}
            onKeyDown={handleZoneKeyDown}
            onClick={() => fileInput.current?.click()}
            role="button"
            tabIndex="0"
            aria-label="Choose or drop a citrus photo"
          >
            <span className="upload-icon" aria-hidden="true">
              <svg viewBox="0 0 32 32">
                <path d="M16 21V5m0 0-6 6m6-6 6 6M6 19v8h20v-8" />
              </svg>
            </span>
            <strong>{selectedFile ? selectedFile.name : 'Drop a photo here'}</strong>
            <span>or browse a file from your device</span>
            <span className="file-hint">Image files · up to 10 MB</span>
            <input
              ref={fileInput}
              className="visually-hidden"
              type="file"
              accept="image/*"
              aria-label="Choose a citrus photo"
              aria-hidden="true"
              tabIndex="-1"
              onClick={(event) => event.stopPropagation()}
              onChange={(event) => {
                chooseFile(event.target.files?.[0])
                event.target.value = ''
              }}
            />
          </div>

          {error && <p className="upload-error" role="alert">{error}</p>}
          <div className="upload-actions">
            <button
              className="button button-secondary"
              type="button"
              onClick={() => fileInput.current?.click()}
            >
              Choose photo
            </button>
            <button
              className="button button-primary"
              type="submit"
              disabled={!selectedFile || loading}
            >
              {loading && <span className="button-spinner" aria-hidden="true" />}
              {loading ? 'Classifying…' : 'Classify photo'}
            </button>
          </div>
          {previewUrl && (
            <div className="upload-preview">
              <img src={previewUrl} alt="Preview of the selected citrus fruit" />
              <div>
                <span>Ready to classify</span>
                <strong>{selectedFile?.name}</strong>
              </div>
            </div>
          )}
        </form>

        <div className="result-column" aria-live="polite">
          {loading ? (
            <div className="panel result-loading" role="status">
              <span className="large-spinner" />
              <strong>Looking closely at the peel…</strong>
              <span>The model is preparing a classification.</span>
            </div>
          ) : result ? (
            <ResultCard result={result} classes={classes} />
          ) : (
            <div className="panel result-placeholder">
              <div className="placeholder-fruit" aria-hidden="true">◌</div>
              <h3>Your result will appear here</h3>
              <p>Select a clear citrus peel photo to see its predicted class and grading recommendation.</p>
            </div>
          )}
        </div>
      </div>
    </>
  )
}

export default Uploader
