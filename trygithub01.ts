
import { GoogleGenAI, Type, GenerateContentResponse } from "@google/genai";
import { InterviewResult, SESSION_1_FEATURES, calculateKanoCategory, KanoFeeling } from '../types';

const getGeminiClient = () => {
    return new GoogleGenAI({ apiKey: process.env.API_KEY as string });
};

// Helper: Wait function
const wait = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

// Helper: Wait for file to be processed by Google (Video processing takes time)
const waitForFileActive = async (ai: GoogleGenAI, fileName: string) => {
    let file = await ai.files.get({ name: fileName });
    while (file.state === 'PROCESSING') {
        // Wait 5 seconds before checking again
        await wait(5000);
        file = await ai.files.get({ name: fileName });
    }
    if (file.state !== 'ACTIVE') {
        throw new Error(`File processing failed. State: ${file.state}`);
    }
    return file;
};

// Legacy Helper: For small files only
const fileToGenerativePart = async (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => {
            const base64String = reader.result as string;
            const base64Data = base64String.split(',')[1];
            resolve(base64Data);
        };
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
};

const cleanJsonString = (text: string): string => {
    if (!text) return "{}";
    let cleaned = text.trim();
    // Remove markdown code blocks if present
    if (cleaned.startsWith('```json')) {
        cleaned = cleaned.replace(/^```json\s*/, '').replace(/\s*```$/, '');
    } else if (cleaned.startsWith('```')) {
        cleaned = cleaned.replace(/^```\s*/, '').replace(/\s*```$/, '');
    }
    return cleaned;
};

export const analyzeInterview = async (file: File, fileId: string, modelName: string = 'gemini-3-pro-preview'): Promise<Partial<InterviewResult>> => {
    const ai = getGeminiClient();
    const mimeType = file.type || 'video/mp4';

    // Construct the feature list string for the prompt
    const featuresList = SESSION_1_FEATURES.map(f => `- "${f.key}": ${f.en} (${f.th})`).join('\n');

    const systemPrompt = `
  You are an **Expert R&D Sensory & Marketing Analyst**.
  Your task is to extract structured data from a product testing interview (Chemical/Sealant industry).

  **CRITICAL LANGUAGE RULE:** 
  1. The output JSON values for 'summary', 'decisionReasoning', 'note', 'match_reason', 'persona_profile' MUST BE in **THAI LANGUAGE (ภาษาไทย)**.
  2. Even if the interviewee speaks English, you must TRANSLATE the insights into Thai.
  
  ---------------------------------------------------------
  **TASK 1: KANO MODEL ANALYSIS (CRITICAL)**
  You MUST analyze the user's satisfaction for ALL 18 features listed below.
  For each feature, deduce:
  1. **Functional (เมื่อมี):** How do they feel if the product HAS this feature? (Like, Expect, Neutral, Tolerate, Dislike)
  2. **Dysfunctional (เมื่อไม่มี):** How do they feel if the product LACKS this feature? (Like, Expect, Neutral, Tolerate, Dislike)
  
  **TARGET FEATURES LIST:**
  ${featuresList}

  *Rule:* If the user doesn't explicitly mention a feature, use your best judgment based on their persona (e.g., a pro user usually EXPECTS 'durability'), OR mark as 'Neutral'/'Neutral' (Indifferent) and note "ไม่ได้พูดถึง".
  
  ---------------------------------------------------------
  **TASK 2: SENSORY BLIND TEST (EXTREME PRIORITY)**
  The user is testing a series of samples. **There are typically 8 to 9 distinct samples.**
  You MUST find data for **ALL** of them. Do not stop early.
  Look for keywords like "ตัวที่ 1", "เบอร์ 2", "ถัดไป", "อันสุดท้าย".
  
  For each sample:
  - **Sample ID:** (e.g., "Sample 1", "Sample 9")
  - **Scores:** Estimate sensory scores (-2 to +2).
  - **Technical Analysis:** Explain the property inferred from their feedback in Thai.

  ---------------------------------------------------------
  ---------------------------------------------------------
  **TASK 3: MARKETING STRATEGY & HABITS**
  Analyze the interview from a Marketer's perspective.
  - **Target Audience:** Who is this product best for? (Thai)
  - **USP:** What is the ONE thing that will make them buy? (Thai)
  - **Hooks:** Catchy phrases to use in ads (Thai).
  - **Channels:** Where would they buy this? (Thai)
  - **Buying/Usage Habits:** Describe in detail how they choose and use products. What factors influence their decision? How do they typically work? (Thai)
  - **Pain Points:** What are their biggest frustrations with current products? (Thai)

  ---------------------------------------------------------
  **TASK 4: BRAND, EXPERT & SEALANT ANALYSIS**
  - **Persona Profile:** Short Title in Thai (e.g., "ช่างฝีมือผู้หลงใหลความสมบูรณ์แบบ").
  - **Sealant Character:** Detailed description of their behavior and technical preferences when using sealant based on their quotes. (Thai)
  - **Lifestyle/Work:** What is their general work/life style?
  - **Unspoken Needs:** Deep insights not explicitly stated.
  - **Sealant Feedback:** Specific feedback on the sealant properties they discussed (e.g., texture, drying time, adhesion). (Thai)
  
  ---------------------------------------------------------
  **TASK 5: TECHNICIAN CLASSIFICATION (NEW & CRITICAL)**
  Classify the user into EXACTLY ONE of these 6 types based on their priorities:
  
  1. **ช่างรับเหมาก่อสร้าง (Construction Contractor)**
     - Focus: Structure, Joints, Durability, Standard Spec.
     - Keywords: ไม่แตก, ไม่พัง, ไม่แก้งาน, ความทนทาน.
  
  2. **ช่างประตู–หน้าต่าง / อลูมิเนียม (Door/Window/Aluminum)**
     - Focus: Frames, Glass, Esthetics, Easy extrusion.
     - Keywords: งานสวย, ยิงลื่น, เส้นสวย, ไม่เหลือง.

  3. **ช่างกระจก (Glass Technician)**
     - Focus: Glass rooms, Façade, Chemical compatibility.
     - Keywords: เข้ากับกระจกได้, ไม่กัด, ไม่ Stain, Neutral.

  4. **ช่างตกแต่งภายใน / บิวท์อิน (Interior/Built-in)**
     - Focus: Finishing, Paintable, Low odor.
     - Keywords: ทาสีทับได้, เนียน, Low VOC, กลิ่นไม่ฉุน.

  5. **ช่างซ่อมบำรุง / ช่างอเนกประสงค์ (Maintenance/Handyman)**
     - Focus: Quick fix, Versatile, All-in-one.
     - Keywords: หลอดเดียวจบ, ใช้งานง่าย, อุดโป๊ว.

  6. **ช่างเฉพาะทาง (Specialist - Roof/Sanitary)**
     - Focus: Extreme durability, Weather resistance.
     - Keywords: ทนแดด, ทนน้ำ, ทนสภาวะรุนแรง.

  *Output:* Return the Thai Name of the type, the Confidence (1-100), and specific keywords they said that match this type.
  `;

    const schema = {
        type: Type.OBJECT,
        properties: {
            summary: { type: Type.STRING },
            decisionReasoning: { type: Type.STRING },
            transcript: { type: Type.STRING },
            session1: {
                type: Type.ARRAY,
                items: {
                    type: Type.OBJECT,
                    properties: {
                        featureKey: { type: Type.STRING },
                        functional: { type: Type.STRING, enum: ['Like', 'Expect', 'Neutral', 'Tolerate', 'Dislike'] },
                        dysfunctional: { type: Type.STRING, enum: ['Like', 'Expect', 'Neutral', 'Tolerate', 'Dislike'] },
                        note: { type: Type.STRING }
                    }
                }
            },
            marketing_analysis: {
                type: Type.OBJECT,
                properties: {
                    target_audience: { type: Type.STRING },
                    unique_selling_point: { type: Type.STRING },
                    pricing_perception: { type: Type.STRING },
                    marketing_channels: { type: Type.ARRAY, items: { type: Type.STRING } },
                    marketing_hooks: { type: Type.ARRAY, items: { type: Type.STRING } },
                    pain_point_solution: { type: Type.STRING }
                }
            },
            technician_profile: {
                type: Type.OBJECT,
                properties: {
                    type: { type: Type.STRING },
                    match_reason: { type: Type.STRING },
                    primary_keywords: { type: Type.ARRAY, items: { type: Type.STRING } },
                    confidence_score: { type: Type.NUMBER }
                }
            },
            qualitative: {
                type: Type.OBJECT,
                properties: {
                    brands: {
                        type: Type.ARRAY,
                        items: {
                            type: Type.OBJECT,
                            properties: {
                                brand: { type: Type.STRING },
                                product: { type: Type.STRING },
                                sentiment: { type: Type.STRING, enum: ['Positive', 'Negative', 'Neutral'] },
                                reason: { type: Type.STRING },
                                comparison_points: { type: Type.ARRAY, items: { type: Type.STRING } },
                                switching_barrier: { type: Type.STRING }
                            }
                        }
                    }
                }
            },
            session2: {
                type: Type.ARRAY,
                items: {
                    type: Type.OBJECT,
                    properties: {
                        sampleId: { type: Type.STRING },
                        note: { type: Type.STRING },
                        technical_analysis: { type: Type.STRING },
                        scores: {
                            type: Type.OBJECT,
                            properties: {
                                hard_soft: { type: Type.NUMBER },
                                smooth_rough: { type: Type.NUMBER },
                                matte_glossy: { type: Type.NUMBER },
                                reflective: { type: Type.NUMBER },
                                cold_warm: { type: Type.NUMBER },
                                elasticity: { type: Type.NUMBER },
                                opacity: { type: Type.NUMBER },
                                tough_ductile: { type: Type.NUMBER },
                                strong_weak: { type: Type.NUMBER },
                                light_heavy: { type: Type.NUMBER },
                            }
                        }
                    }
                }
            },
            session3: {
                type: Type.OBJECT,
                nullable: true,
                properties: {
                    ease_of_extrusion: { type: Type.NUMBER },
                    smooth_finishing: { type: Type.NUMBER },
                    initial_tack: { type: Type.NUMBER },
                    odor_acceptability: { type: Type.NUMBER },
                    note: { type: Type.STRING }
                }
            },
            expert_insights: {
                type: Type.OBJECT,
                properties: {
                    persona_profile: { type: Type.STRING },
                    sealant_personality: { type: Type.STRING },
                    professional_lifestyle: { type: Type.STRING },
                    expertise_level: { type: Type.STRING },
                    unspoken_needs: { type: Type.ARRAY, items: { type: Type.STRING } },
                    rd_roadmap_suggestions: { type: Type.ARRAY, items: { type: Type.STRING } },
                    critical_quotes: { type: Type.ARRAY, items: { type: Type.STRING } }
                }
            }
        },
        required: ["summary", "session1", "session2", "expert_insights", "marketing_analysis", "technician_profile"]
    };

    try {
        let contentPart;

        // CHECK FILE SIZE: If > 20MB, use File API (Upload) instead of Inline Base64
        // This prevents browser memory crashes with large/long files.
        if (file.size > 20 * 1024 * 1024) {
            // Option A: Large File -> Upload to Gemini Files
            try {
                const uploadResult = await ai.files.upload({
                    file: file,
                    config: {
                        displayName: file.name,
                        mimeType: mimeType
                    }
                });

                // Wait for video/audio processing to complete (ACTIVE state)
                // Access .name directly on uploadResult as it returns the file object
                const processedFile = await waitForFileActive(ai, uploadResult.name);

                contentPart = {
                    fileData: {
                        mimeType: processedFile.mimeType,
                        fileUri: processedFile.uri
                    }
                };
            } catch (uploadError: any) {
                console.error("Upload failed:", uploadError);
                throw new Error(`Large file upload failed: ${uploadError.message}. Try checking connection or file format.`);
            }

        } else {
            // Option B: Small File -> Use Inline Base64 (Faster for small clips)
            const base64Data = await fileToGenerativePart(file);
            contentPart = { inlineData: { mimeType, data: base64Data } };
        }

        // RETRY LOGIC FOR RATE LIMITS (429 / Resource Exhausted)
        let retries = 5;
        let delay = 3000; // Start with 3 seconds
        let response: GenerateContentResponse | null = null;

        while (retries > 0) {
            try {
                response = await ai.models.generateContent({
                    model: 'gemini-3-pro-preview',
                    contents: {
                        parts: [contentPart, { text: systemPrompt }]
                    },
                    config: {
                        responseMimeType: "application/json",
                        responseSchema: schema,
                        temperature: 0.2
                    }
                });

                // If successful, break loop
                break;
            } catch (apiError: any) {
                const errMsg = apiError.toString().toLowerCase();
                const isRetryable =
                    errMsg.includes('429') ||
                    errMsg.includes('503') ||
                    errMsg.includes('overloaded') ||
                    errMsg.includes('resource') ||
                    errMsg.includes('quota') ||
                    errMsg.includes('exhausted');

                if (isRetryable && retries > 1) {
                    console.warn(`Model overloaded or Rate limit hit. Retrying in ${delay}ms... (Attempts left: ${retries - 1})`);
                    await wait(delay);
                    delay *= 2.5; // Slightly more aggressive backoff for 503s
                    retries--;
                } else {
                    // If not a rate limit error, or we ran out of retries, throw it
                    throw apiError;
                }
            }
        }

        if (!response || !response.text) {
            throw new Error("No response text from Gemini");
        }

        const jsonString = cleanJsonString(response.text);
        const json = JSON.parse(jsonString);

        const processedSession1 = SESSION_1_FEATURES.map(feature => {
            const found = (json.session1 || []).find((item: any) => item.featureKey === feature.key);
            if (found) {
                return {
                    ...found,
                    category: calculateKanoCategory(found.functional as KanoFeeling, found.dysfunctional as KanoFeeling)
                };
            }
            return {
                featureKey: feature.key,
                functional: 'Neutral',
                dysfunctional: 'Neutral',
                category: 'I',
                note: 'ไม่พบข้อมูลในบทสัมภาษณ์'
            };
        });

        return {
            status: 'completed',
            summary: json.summary,
            decisionReasoning: json.decisionReasoning,
            transcript: json.transcript,
            session1: processedSession1,
            qualitative: json.qualitative,
            session2: json.session2 || [],
            session3: json.session3 || { ease_of_extrusion: 0, smooth_finishing: 0, initial_tack: 0, odor_acceptability: 0, note: "ไม่ได้ระบุ" },
            expert_insights: json.expert_insights,
            marketing_analysis: json.marketing_analysis,
            technician_profile: json.technician_profile
        };
    } catch (error: any) {
        console.error("Gemini Error:", error);
        return { status: 'error', errorMessage: error.message };
    }
};
