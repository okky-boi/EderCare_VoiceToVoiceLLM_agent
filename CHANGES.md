# Perubahan pada `/api/process-audio` Endpoint

## Summary

Endpoint `/api/process-audio` telah diubah untuk menerima dan mengembalikan **binary audio stream** (octet-stream) daripada menggunakan file URL dengan bentuk teks response.

## Perubahan Detail

### 1. **Endpoint Input**

-   **Sebelumnya**: Menerima `UploadFile` (multipart/form-data)
-   **Sekarang**: Menerima `Request` dengan body berupa binary audio stream (application/octet-stream)

### 2. **Endpoint Output**

-   **Sebelumnya**: JSON response (`ChatResponse`) dengan `audio_url` pointing ke file lokal
-   **Sekarang**: Binary audio stream (MP3) langsung dengan metadata di HTTP headers

### 3. **File Management**

-   **Sebelumnya**: Audio response disimpan ke `audio_cache/` dan dihapus secara manual
-   **Sekarang**: Audio response di-generate ke memory (`io.BytesIO`) tanpa menyimpan ke disk

### 4. **Response Headers**

Response sekarang menyertakan metadata dalam HTTP headers:

-   `X-Transcription`: Teks hasil transkripsi dari input audio
-   `X-Response`: Teks respons dari LLM
-   `X-Response-Type`: Tipe respons ("conversation" atau "action")
-   `X-Latency-Ms`: Latency dalam milidetik
-   `X-Function-Call`: JSON function call (jika ada)
-   `Content-Disposition`: Filename untuk download

## Files Modified

### 1. `src/api/server.py`

-   ✅ Updated imports: Tambah `Request`, `StreamingResponse`, `io`
-   ✅ Added `generate_audio_bytes()` method ke MochiAPI class
-   ✅ Changed endpoint dari `response_model=ChatResponse` menjadi `@app.post("/api/process-audio")`
-   ✅ Changed implementation untuk menggunakan binary stream

### 2. `src/core/tts_service.py`

-   ✅ Updated imports: Tambah `io`
-   ✅ Added `generate_bytes()` method: Generate audio bytes ke memory tanpa menyimpan file

## Usage

### Request Format

```bash
curl -X POST http://localhost:8000/api/process-audio \
  -H "Content-Type: application/octet-stream" \
  --data-binary @audio.wav
```

### Response

-   Content-Type: `audio/mpeg`
-   Body: Binary MP3 audio stream
-   Headers: Metadata dalam custom headers (X-\*)

## Benefits

✅ **Lebih efisien**: Tidak ada file yang tersimpan di disk  
✅ **Streaming real-time**: Audio bisa langsung diplay tanpa menunggu full response  
✅ **Metadata lengkap**: Info transkripsi dan response tersedia di headers  
✅ **Cleaner client**: Client hanya perlu menangani binary stream, bukan URL  
✅ **Automatic cleanup**: Tidak ada files numpuk di `audio_cache/`
