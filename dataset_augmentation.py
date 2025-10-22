import pandas as pd
import os
from load_dotenv import load_dotenv
import google.generativeai as genai
import time

load_dotenv()

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

model = genai.GenerativeModel('gemini-flash-latest')

prompt = """Sei un annotatore per un task di classificazione. In input riceverai la bio di un utente Twitter e alcuni suoi tweet. 
Il tuo compito è decidere se l'utente in questione fa parte o meno della comunità LGBT. 
Basa la tua scelta sul modo di scrivere, sul contenuto della bio e dei tweet, e su tutti i fattori che ritieni rilevanti.

L'input che ti fornirò sarà nel formato TWEET - BIO, ad esempio:

"Mi sento così libera da quando ho fatto coming out 🌈 || amo la mia ragazza" - "studentessa di filosofia, lei/she, femminista, attivista lgbtqia+"

L'output che voglio è semplicemente un numero:
0 se l'utente NON appartiene alla comunità LGBT,
1 se invece appartiene alla comunità.

Esempio di interazione:

INPUT:
"fuck gender rules and the rules of society || bts || exo" - "pansexual, genderqueer and polyamorous 🏳️‍🌈 || she/her || unito dams"

OUTPUT: 1

INPUT:
"{}" - "{}"

OUTPUT:
"""



ita = pd.read_csv("dataset/train_it.csv")

lgbt = []
c = 0
for text,bio in zip(ita["text"],ita["bio"]):
    c+=1
    print(f"{c} / {len(ita)}")
    if str(bio) == "nan":
        bio = '""'
    format_prompt = prompt.format(text, bio)
    print(f"{text} - {bio}")

    try:
        response = model.generate_content(format_prompt)
        # Stampa il testo della risposta
        answer = response.text
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
    print("--"*20)
    time.sleep(7)


ita["lgbt"] = lgbt

print(ita.head())

ita.to_csv("augmented.csv")