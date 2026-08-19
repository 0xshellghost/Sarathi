export async function streamAnalyzeRequest(
  userInput: string,
  onToken: (text: string) => void,
  onFormSchema: (schema: any) => void
) {
  try {
    const res = await fetch("http://localhost:8000/api/v1/action/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_input: userInput }),
    })

    if (!res.body) throw new Error("No response body")

    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ""

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split("\n")
      
      // Hold onto the last incomplete line until the next chunk arrives
      buffer = lines.pop() || ""

      for (const line of lines) {
        const trimmed = line.trim()
        if (!trimmed.startsWith("data:")) continue
        
        const raw = trimmed.replace(/^data:\s*/, "")
        if (!raw) continue

        try {
          const data = JSON.parse(raw)
          
          if (data.type === "token") {
            onToken(data.text)
          } else if (data.type === "form_request") {
            onFormSchema(data.schema)
          }
        } catch {
          // Silently skip malformed JSON chunks
        }
      }
    }
  } catch (error) {
    console.error("Stream connection failed:", error)
  }
}