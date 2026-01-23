# Ethics & Responsible Use

This project performs **facial expression recognition** (emotion classification) from face images.

## Important limitations

- **Emotions are not ground truth.** Facial expressions are an imperfect proxy for internal emotional state.
- **Bias & fairness:** Performance can vary across demographics (age, gender presentation, skin tone), lighting, camera quality, and cultural expression patterns.
- **Context matters:** Expressions depend heavily on context and can be ambiguous.

## Appropriate use

- ✅ Demos, UX prototyping, HCI research, robotics/HRI experiments with consent.
- ✅ Non-critical analytics where errors are acceptable.

## Inappropriate / high-stakes use

- ❌ Employment screening, hiring decisions, school discipline, medical diagnosis, policing, or any decision that impacts rights, benefits, or safety.

## Privacy

- Do not store or transmit user images without explicit consent.
- Prefer on-device inference when possible.
- If deploying an API, secure it (auth, TLS) and implement retention/deletion policies.
