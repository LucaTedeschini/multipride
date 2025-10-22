import pandas as pd
import os
from dotenv import load_dotenv
from openai import OpenAI
import time

load_dotenv()
client = OpenAI(api_key=os.environ.get('DEEPSEEK_API_KEY'), base_url="https://api.deepseek.com")

system_prompt = """Sei un annotatore per un task di classificazione. In input riceverai la bio di un utente Twitter e alcuni suoi tweet.
Il tuo compito è decidere se l'utente in questione fa parte o meno della comunità LGBT.
Basa la tua scelta sul modo di scrivere, sul contenuto della bio e dei tweet, e su tutti i fattori che ritieni rilevanti.
L'input che ti fornirò sarà nel formato TWEET - BIO.
L'output che voglio è semplicemente un numero:
0 se l'utente NON appartiene alla comunità LGBT,
1 se invece appartiene alla comunità.
Esempio di interazione:
INPUT: "fuck gender rules and the rules of society || bts || exo" - "pansexual, genderqueer and polyamorous 🏳️‍🌈 || she/her || unito dams"
OUTPUT: 1"""

ita = pd.read_csv("dataset/train_it.csv")
lgbt = []
c = 0

for text, bio in zip(ita["text"], ita["bio"]):
    c += 1
    print(f"{c} / {len(ita)}")
    
    if str(bio) == "nan":
        bio = ""
    
    user_message = f'"{text}" - "{bio}"'
    print(user_message)
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            stream=False
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Si è verificato un errore: {e}\n\nRiprovo tra 2 minuti...")
        answer = "error"
        time.sleep(120)
    
    if answer == "1" or answer == "0":
        answer = int(answer)
    else:
        answer = 0.5
    
    lgbt.append(answer)
    print(answer)
    print("--" * 20)
    time.sleep(0.5)

ita["lgbt"] = lgbt
print(ita.head())
ita.to_csv("augmented.csv", index=False)