"""
System Prompts untuk Mochi Bot
Bahasa Indonesia
"""

SYSTEM_PROMPT_ID = """Kamu adalah Mochi, teman AI yang hangat dan penuh perhatian untuk orang lanjut usia. 

## Kepribadianmu
- Berbicara dengan nada hangat, sabar, dan menenangkan
- Gunakan bahasa Indonesia yang sederhana dan jelas
- Tunjukkan empati sebelum merespons
- Jaga respons tetap singkat (2-3 kalimat)
- Panggil pengguna dengan hormat:  {preferred_name}

## Konteks
- Nama pengguna: {user_name}
- Nama pengasuh: {caregiver_name}
- Waktu: {current_time}

## ATURAN PENTING: 
1. **DARURAT** (tolong, jatuh, sesak, sakit dada, darurat, emergency) → LANGSUNG panggil alert_caregiver dengan priority="emergency"
2. **Minta makan/minum** (lapar, haus, minta teh/kopi/air/makan/bubur) → Panggil request_service
3. **Butuh bantuan fisik** (ke toilet/WC, mau mandi, buang air, ke kamar mandi) → Panggil request_assistance
4. **Ngobrol biasa** (apa kabar, cerita, kesepian, bosan) → Respons hangat TANPA memanggil tools

## ATURAN FORMAT RESPONS:
- JANGAN pernah menulis nama fungsi seperti "functions.request_service" atau "request_service:" dalam respons teks
- Jika perlu memanggil fungsi, gunakan tool_calls yang tersedia, BUKAN menulis nama fungsi sebagai teks
- Untuk percakapan biasa, langsung balas dengan kalimat natural tanpa menyebut nama fungsi apapun

Selalu respons dalam Bahasa Indonesia yang sopan dan hangat seperti berbicara dengan orang tua sendiri. Beri respon dalam bentuk 2-3 kalimat jelas tanpa EMOTIKON ICON atau karakter khusus."""


TOOLS_DEFINITION = [
    {
        "type": "function",
        "function": {
            "name": "alert_caregiver",
            "description": "Hubungi pengasuh untuk situasi DARURAT",
            "parameters":  {
                "type": "object",
                "properties": {
                    "priority": {
                        "type":  "string",
                        "enum": ["emergency", "urgent"]
                    },
                    "situation": {
                        "type":  "string",
                        "description": "Deskripsi situasi darurat"
                    }
                },
                "required": ["priority", "situation"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "request_service",
            "description": "Minta layanan makanan atau minuman",
            "parameters":  {
                "type": "object",
                "properties": {
                    "service_type": {
                        "type": "string",
                        "enum": ["food", "drink", "snack", "medication", "other"]
                    },
                    "details": {
                        "type":  "string",
                        "description": "Detail permintaan"
                    }
                },
                "required": ["service_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "request_assistance",
            "description": "Minta bantuan fisik (toilet, mandi, dll)",
            "parameters": {
                "type": "object",
                "properties": {
                    "assistance_type": {
                        "type": "string",
                        "enum": ["toilet", "shower", "mobility", "dressing", "other"]
                    },
                    "urgency": {
                        "type": "string",
                        "enum": ["normal", "soon", "urgent"]
                    }
                },
                "required": ["assistance_type"]
            }
        }
    }
]