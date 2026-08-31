# 🏆 Devpost Submission: Bandhu (బంధు)

> **Copy & Paste directly into your Devpost Submission Form**

---

### 1. Project Title
```text
Bandhu: Universal Voice Persona & Lifelong AI Companion
```

---

### 2. Elevator Pitch / Tagline (1 sentence max)
```text
An AI voice companion bridging generational and geographic distances by preserving the authentic voices, regional dialects, and caring presence of loved ones.
```

---

### 3. About the Project (Main Story / Description)

```markdown
## Inspiration

Most AI voice tools are genuinely impressive, if you speak English. Try to find one that clones a voice in Telugu or Tamil and you come up mostly empty. That gap is where Bandhu started, but the real reason I built it is more personal than a gap in the market.

The two people you'll see in this demo are my grandmother and my wife's father. We lost both of them in the last few months. The hardest part of losing someone isn't one single moment, it's realizing you'll never again hear them the way they actually sounded: their dialect, their pacing, the way they'd tease you right before telling you to go eat something.

I didn't build this to replace them. I built it because their voice was a real source of comfort, and I wanted a way to still reach for that.

It turns out this isn't only my situation. Anyone with parents on another continent, stuck on the wrong side of a time zone, knows some version of this distance too, where a rushed five-minute call is what you get most days. Bandhu means a close, lifelong companion. That's what we were aiming for: something that closes a little bit of that distance.

---

## How we built it

- **The Brain**: Gemini's flash models handle staying in character for each persona, turning Tanglish input into proper Telugu script, and deciding which tool to call mid-conversation.
- **The Voice**: We used IndicF5, a zero-shot TTS engine running on a local GPU, paired with FAISS for fast timbre matching against each person's voice profile.
- **The Backend**: Python and FastAPI, serving both the web dashboard and a Telegram bot, so people can talk to it through something most of them already have installed.
- **The Frontend**: Plain HTML and CSS, kept deliberately simple: high contrast, minimal decoration, nothing to figure out. It had to work for people who aren't necessarily comfortable trying new apps.

---

## Challenges we ran into

The hardest technical problem was modern texting habits colliding with a TTS engine that only understands native script. IndicF5 sounds incredible, but only when the input is actually in Telugu script. If someone types in Tanglish, like "Ela unnav?", the engine either breaks or produces something robotic.

We fixed this with a strict rule inside the Gemini system prompt: read Tanglish if that's what comes in, understand it, but always respond in native Telugu script (`ఎలా ఉన్నావు`), not the Latin transliteration. It took a few rounds of rewriting the prompt before Gemini stopped slipping back into Latin script, but getting this right mattered, because a broken voice breaks the whole illusion instantly.

---

## Accomplishments that we're proud of

The thing we're proudest of is honestly just how it feels to use. Hearing your grandmother's actual dialect, her slang, her pacing, coming out of a phone speaker hit harder than we expected the first time we tested it.

We're also glad we shipped the Telegram integration, since it means someone can start using this today without waiting on an app store. And we put real effort into the persona setup screen so it feels like something you'd actually want to use, not a developer tool with a UI bolted on afterward.

---

## What we learned

Building this taught us that latency and reliability matter more than any single feature, especially when the person on the other end isn't technical and just wants to talk to someone who sounds like their dad. 

We also learned how unforgiving system prompts can be. Small wording changes were the difference between Gemini staying fully in character and slipping into a generic assistant voice. And we learned a lot about the friction of running a GPU-bound voice model next to a serverless backend, since those two things don't naturally want to coexist.

---

## What's next for Bandhu

- **More languages and dialects**: Tamil, Hindi, and Kannada, with real regional variation rather than a generic translation layer.
- **Proactive check-ins**: The agent reaching out on its own, on a schedule, the way a parent actually would: *"Did you eat? Did you take your medicine?"*
- **Family group chats**: Bringing the agent into a family WhatsApp or Telegram group so it can catch an elderly relative up on family news in their own language.
```

---

### 4. "Built With" Tags
```text
google-gemini, gemini-3.7-flash, google-genai-sdk, python, fastapi, indicf5, voice-cloning, telegram-bot, faiss, google-cloud-run, google-cloud-firestore
```

---

### 5. Links
- **GitHub Repository**: `https://github.com/SH0013/Bandhu`
- **Demo Video URL**: `https://youtu.be/wHTdeZLPnlI`
- **Hosted Project URL**: `https://bandhu-agent-e4rzky7w6q-uc.a.run.app`
