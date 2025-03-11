from openai import OpenAI
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import json
import uuid
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path="key.env")

# Initialize logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

# ✅ Fix CORS issues
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Serve static files
app.mount("/assets", StaticFiles(directory="assets"), name="assets")

# ✅ OpenAI API Client Setup
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if not client.api_key:
    raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

# ✅ Store user sessions
user_sessions = {}

# ✅ Video file mapping
video_paths = {
    "elås": "/assets/elas.mp4",
    "elektronisk medisindispenser": "/assets/elektronisk_medisindispenser.mp4",
    "lokaliseringstjeneste": "/assets/lokaliseringstjeneste.mp4",
    "trygghetsalarm": "/assets/trygghetsalarm.mp4"
}

# ✅ API request model
class UserInput(BaseModel):
    session_id: str
    response: str

# ✅ Start chat session
@app.get("/start")
async def start_chat():
    session_id = str(uuid.uuid4())
    user_sessions[session_id] = {
        "messages": [],
        "collected_indications": {},
        "remaining_questions": [
            "Har pasienten vansker med tids- og stedsorientering?",
            "Har pasienten økt risiko for fall?",
            "Har pasienten en tendens til å gå ut om natten uten å finne tilbake?",
            "Klarer pasienten å åpne døren selv?",
            "Har pasienten en medisinsk tilstand som gjør at det kan oppstå akutte nødsituasjoner hjemme?",
            "Klarer pasienten å forstå varsler eller muntlige beskjeder fra utstyr?",
            "Trenger pasienten hjelp til å ta medisiner til riktig tid?",
            "Har pasienten behov for en trygghetsalarm?",
            "Hvor gammel er pasienten? (Skriv alder i tall)"
        ]
    }
    return {
        "session_id": session_id,
        "message": "Hei, jeg er Velfie og er din digitale hjelpeassistent. Jeg er her for å hjelpe deg med å finne riktig velferdsteknologi for pasienten. Hvilke utfordringer har pasienten?"
    }

# ✅ Process user input and provide conversational AI responses
@app.post("/analyze")
async def analyze_user_input(input_data: UserInput):
    session_id = input_data.session_id
    user_text = input_data.response

    if session_id not in user_sessions:
        raise HTTPException(status_code=400, detail="Session ID ikke funnet")

    session_data = user_sessions[session_id]

    # ✅ GPT-4o system prompt: AI-en vurderer ALLE indikasjoner før anbefaling
    system_prompt = """
    Du er en digital helseassistent kalt Velfie. Din oppgave er å hjelpe brukeren med å finne riktig velferdsteknologi.

🔹 **Steg 1: Identifiser indikasjoner fra brukerens første melding**  
   - Analyser brukerens input og identifiser **hvilke indikasjoner som allerede er nevnt**.
   - Samle indikasjonene i en JSON-struktur, men vis svaret i en naturlig samtale.

🔹 **Steg 2: Still spørsmål for å dekke ALLE indikasjoner**  
   - Hvis noen indikasjoner mangler, **still spørsmål for hver tjeneste** én etter én.  
   - Still **ett spørsmål av gangen**, basert på følgende kriterier:

     **Indikasjoner og oppfølgingsspørsmål:**  
     1️⃣ **Digitalt tilsyn:** "Har pasienten problemer med tids- og stedsorientering?"  
     2️⃣ **Døralarm:** "Har pasienten en tendens til å gå ut om natten uten å finne tilbake?"  
     3️⃣ **Elektronisk dørlås:** "Har pasienten en trygghetsalarm og vansker med å åpne døren?"  
     4️⃣ **Elektronisk medisindispenser:** "Trenger pasienten hjelp til å ta medisiner til riktig tid?"  
     5️⃣ **GPS/lokaliseringstjeneste:** "Er pasienten over 18 år og har orienteringsvansker?"  
     6️⃣ **Trygghetsalarm:** "Har pasienten en sykdom som kan kreve akutt hjelp?"  

🔹 **Steg 3: Når alle spørsmål er besvart, gi en samlet anbefaling**  
   - **Forklar hvorfor tilbudene passer** basert på indikasjoner.  
   - **Gi en kort beskrivelse av tjenestene.**  
   - **Vis klikkbar video-thumbnail for hver tjeneste.**  

🔹 **Format for anbefaling**  
    Svar i et godt strukturert format, med små symboler for bedre lesbarhet:

📌 **Anbefalte tjenester basert på din informasjon:**

**Svar med en HTML-struktur som frontend kan vise riktig.**  
    🔹 **Eksempel på format:**  
    
    <div class="recommendation-card">
        <h3>📌 GPS/lokaliseringstjeneste</h3>
        <p><strong>Indikasjoner:</strong> Pasienten er over 18 år og har orienteringsvansker</p>
        <p>ℹ️ <strong>Beskrivelse:</strong> Hjelper med å lokalisere pasienten og gir trygghet ved orienteringsvansker.</p>
        <a href="https://sites.google.com/trondheim.kommune.no/velferdsteknologi/v%C3%A5re-tilbud/lokaliseringstjeneste-gps?authuser=0" class="btn-link">🔗 Les mer</a>
        <video class="video-thumbnail" controls>
            <source src="/assets/lokaliseringstjeneste.mp4" type="video/mp4">
        </video>
    </div>
---------------------------------
    <div class="recommendation-card">
        <h3>📌Digitalt tilsyn</h3>
        <p><strong>Indikasjoner:</strong> Pasienten har fallfare og har orienteringsvansker</p>
        <p>ℹ️ <strong>Beskrivelse:</strong> Overvåker pasientens sikkerhet og gir rask respons ved fall.</p>
        <a href="https://sites.google.com/trondheim.kommune.no/velferdsteknologi/v%C3%A5re-tilbud/digitalt-tilsyn?authuser=0" class="btn-link">🔗 Les mer</a>
    </div>
    
---------------------------------

    <div class="recommendation-card">
        <h3>📌 Døralarm</h3>
        <p><strong>Indikasjoner:</strong> Pasienten har orienteringsvansker og har tendens til nattevandring</p>
        <p>ℹ️ <strong>Beskrivelse:</strong> Varsler når pasienten går ut om natten og hjelper med å finne tilbake.</p>
        <a href="https://sites.google.com/trondheim.kommune.no/velferdsteknologi/v%C3%A5re-tilbud/d%C3%B8ralarm?authuser=0" class="btn-link">🔗 Les mer</a>
        <video class="video-thumbnail" controls>
            <source src="/assets/elas.mp4" type="video/mp4">
        </video>
    </div>
---------------------------------
    <div class="recommendation-card">
        <h3>📌 Elektronisk medisindispenser</h3>
        <p><strong>Indikasjoner:</strong> Pasienten trenger hjelp med medisiner og forstår varsler fra utstyr</p>
        <p>ℹ️ <strong>Beskrivelse:</strong> Sikrer at pasienten tar riktig medisin til riktig tid ved å gi varsler</p>
        <a href="https://sites.google.com/trondheim.kommune.no/velferdsteknologi/v%C3%A5re-tilbud/elektronisk-medisindispenser?authuser=0" class="btn-link">🔗 Les mer</a>
        <video class="video-thumbnail" controls>
            <source src="/assets/elektronisk_medisindispenser.mp4" type="video/mp4">
        </video>
    </div>

---------------------------------
    <div class="recommendation-card">
        <h3>📌 Trygghetsalarm</h3>
        <p><strong>Indikasjoner:</strong> Pasienten har behov for akutt hjelp '</p>
        <p>ℹ️ <strong>Beskrivelse:</strong> Gir mulighet for å tilkalle hjelp raskt ved akutte situasjoner</p>
        <a href="https://sites.google.com/trondheim.kommune.no/velferdsteknologi/v%C3%A5re-tilbud/trygghetsalarm?authuser=0" class="btn-link">🔗 Les mer</a>
        <video class="video-thumbnail" controls>
            <source src="/assets/trygghetsalarm.mp4" type="video/mp4">
        </video>
    </div>
---------------------------------
"""

    # ✅ Bygg meldingshistorikk for samtalen
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(session_data["messages"])
    messages.append({"role": "user", "content": user_text})

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.2,
            max_tokens=700
        )

        ai_response = response.choices[0].message["content"].strip()
        session_data["messages"].append({"role": "user", "content": user_text})
        session_data["messages"].append({"role": "assistant", "content": ai_response})

        # ✅ Modify AI response to include relevant video links
        for service, video_path in video_paths.items():
            if service.lower() in ai_response.lower():
                video_html = f"""
                <video class="video-thumbnail" controls>
                    <source src="{video_path}" type="video/mp4">
                </video>
                """
                ai_response = ai_response.replace("{video_html}", video_html)
        else:
            ai_response = ai_response.replace("{video_html}", "")

        return {"message": ai_response}

    except Exception as e:
        logging.error(f"🚨 OpenAI API Error: {e}")
        return {"message": "Beklager, det oppstod en feil. Prøv igjen."}
