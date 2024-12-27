package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"html/template"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"strings"
)

const TMPL_INDEX = `
<html>
<h1>{{.Title}}</h1>
<form action="/" method="post">
<input type="text" name="query"/>
<input type="submit" value="Query"/>
</form>

<h2>Response</h2>
{{if .Response}}
<p>Commentary: {{.Response.Commentary}}</p>

References:
<ul>
{{range .Response.Citations}}
<li>{{.Claim}} {{range .References}}<a href="/references/{{.File}}">{{.File}}</a>{{end}}</li>
{{end}}
</ul>

{{else}}
<p>Submit a query to get a response.</p>
{{end}}
</html>
`

const TMPL_PROMPT_EXPAND_QUERY = `
INSTRUCTIONS

Expand the following QUERY. Provide 10 additional queries you would use to
diversify your knowledge on the topic.

QUERY

{{.Query}}

JSON RESPONSE TEMPLATE

{
	"queries": ["<query 1>", "<query 2>", "<query 3>"]
}
`

const TMPL_PROMPT_RESPONSE = `
INSTRUCTIONS

Respond to the following QUERY using REFERENCES below. Cite the a file in the
REFERENCES that contains the fact used. Provide uncited commentary separately.
If NO REFERENCES are provided, it is impossible to provide citations.

QUERY

{{.Query}}

REFERENCES

{{range .References}}
- File: {{.File}}
- Exerpt: {{.Exerpt}}
{{end}}

JSON RESPONSE TEMPLATE

{
	"commentary": "<summary commentary without citations>",
	"citations": [
		{
			"claim": "<summary claim 1>",
			"references": [
				{"exerpt": "<exerpt>", "file": "<file-name-1.txt>"},
				{"exerpt": "<exerpt>", "file": "<file-name-2.txt>"},
				{"exerpt": "<exerpt>", "file": "<file-name-3.txt>"},
			]
		},
	]
}
`

var (
	openAIAPIKeyEnv string = "OPENAI_API_KEY"
	openAIAPIKey    string

	openAILanguageModelEnv     string = "OPENAI_LANGUAGE_MODEL"
	openAILanguageModelDefault string = "gpt-4o"
	openAILanguageModel        string

	openAIEmbeddingModelEnv     string = "OPENAI_EMBEDDING_MODEL"
	openAIEmbeddingModelDefault string = "text-embedding-3-large"
	openAIEmbeddingModel        string

	openAICompletionURL          string = "https://api.openai.com/v1/chat/completions"
	openAICompletionSystemPrompt string = "You are a helpful assistant."

	openAIEmbeddingURL string = "https://api.openai.com/v1/embeddings"

	similarityThreshold float64 = 0.5

	references     []Reference
	referencesPath string = "references"

	templates map[string]*template.Template = make(map[string]*template.Template)
)

type Reference struct {
	File      string    `json:"file"`
	Exerpt    string    `json:"exerpt"`
	Embedding []float64 `json:"embedding,omitempty"`
}

type QueryExpansionResponse struct {
	Queries []string `json:"queries"`
}

type Citation struct {
	Claim      string      `json:"claim"`
	References []Reference `json:"references"`
}

type CitedResponse struct {
	Commentary string     `json:"commentary"`
	Citations  []Citation `json:"citations"`
}

type CompletionRequest struct {
	Model          string              `json:"model"`
	Messages       []map[string]string `json:"messages"`
	ResponseFormat map[string]string   `json:"response_format"`
}

func NewCompletionRequest() *CompletionRequest {
	return &CompletionRequest{
		Model:          openAILanguageModel,
		ResponseFormat: map[string]string{"type": "json_object"},
	}
}

func (r *CompletionRequest) AddMessage(role, content string) {
	r.Messages = append(r.Messages, map[string]string{"role": role, "content": content})
}

type CompletionChoice struct {
	Message map[string]string `json:"message"`
}

type CompletionResponse struct {
	Choices []CompletionChoice `json:"choices"`
}

func NewCompletionResponse() *CompletionResponse {
	return &CompletionResponse{}
}

func (r *CompletionResponse) Content() string {
	return r.Choices[0].Message["content"]
}

type EmbeddingRequest struct {
	Model string `json:"model"`
	Input string `json:"input"`
}

func NewEmbeddingRequest(text string) *EmbeddingRequest {
	return &EmbeddingRequest{
		Model: openAIEmbeddingModel,
		Input: text,
	}
}

type EmbeddingData struct {
	Embedding []float64 `json:"embedding"`
}

type EmbeddingResponse struct {
	Data []EmbeddingData `json:"data"`
}

func (r *EmbeddingResponse) Embedding() []float64 {
	return r.Data[0].Embedding
}

func NewEmbeddingResponse() *EmbeddingResponse {
	return &EmbeddingResponse{}
}

func mustGetEnv(key string) string {
	if value, exists := os.LookupEnv(key); exists {
		return value
	}

	log.Fatalln("required environment variable not set:", key)

	return ""
}

func getEnv(key, defaultValue string) string {
	if value, exists := os.LookupEnv(key); exists {
		return value
	}

	return defaultValue
}

func toJSON(v interface{}) []byte {
	json, err := json.Marshal(v)

	if err != nil {
		log.Fatalln("failed to marshal", err)
	}

	return json
}

func fromJSON(data []byte, v interface{}) interface{} {
	err := json.Unmarshal(data, v)

	if err != nil {
		log.Fatalln("failed to unmarshal", err)
	}

	return v
}

func doPost(url string, body []byte) []byte {
	request, err := http.NewRequest("POST", url, bytes.NewBuffer(body))
	if err != nil {
		log.Fatalf("failed to create request: %v", err)
	}

	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Authorization", fmt.Sprintf("Bearer %s", openAIAPIKey))

	client := &http.Client{}
	response, err := client.Do(request)
	if err != nil {
		log.Fatalf("failed to send request: %v", err)
	}
	defer response.Body.Close()

	if response.StatusCode != http.StatusOK {
		responseBodyBytes, _ := io.ReadAll(response.Body)
		log.Fatalf("unexpected status code: %v, body: %s", response.StatusCode, string(responseBodyBytes))
	}

	responseBodyBytes, err := io.ReadAll(response.Body)
	if err != nil {
		log.Fatalf("failed to read response body: %v", err)
	}

	return responseBodyBytes
}

func LoadTemplate(name string, text string) *template.Template {
	if tmpl, ok := templates[name]; ok {
		return tmpl
	}

	tmpl, err := template.New(name).Parse(text)
	if err != nil {
		log.Fatalf("failed to parse prompt template: %v", err)
	}

	templates[name] = tmpl

	return tmpl
}

func RenderTemplate(name string, data map[string]interface{}) []byte {
	writer := &strings.Builder{}
	err := templates[name].Execute(writer, data)
	if err != nil {
		log.Fatalf("failed to execute template: %v", err)
	}
	return []byte(writer.String())
}

func CalculateSimilarity(a, b []float64) float64 {
	var dotProduct float64
	var aMagnitude float64
	var bMagnitude float64

	for i := 0; i < len(a); i++ {
		dotProduct += a[i] * b[i]
		aMagnitude += a[i] * a[i]
		bMagnitude += b[i] * b[i]
	}

	aMagnitude = math.Sqrt(aMagnitude)
	bMagnitude = math.Sqrt(bMagnitude)

	return dotProduct / (aMagnitude * bMagnitude)
}

func LoadReferences(path string) {
	files, err := os.ReadDir(path)
	if err != nil {
		log.Fatalf("failed to read references directory: %v", err)
	}

	for _, file := range files {
		if file.IsDir() {
			continue
		}

		content, err := os.ReadFile(path + "/" + file.Name())
		if err != nil {
			log.Fatalf("failed to read file %s: %v", file.Name(), err)
		}

		excerpt := string(content)
		embedding := GenerateEmbedding(excerpt)

		reference := Reference{
			File:      file.Name(),
			Exerpt:    excerpt,
			Embedding: embedding,
		}

		references = append(references, reference)
	}
}

func FindReference(query string) []Reference {
	var embedding = GenerateEmbedding(query)
	var result []Reference

	for _, reference := range references {
		// TODO(optim) this is O(n^2) and could be optimized
		for _, r := range result {
			if r.File == reference.File {
				continue
			}
		}

		if CalculateSimilarity(embedding, reference.Embedding) > similarityThreshold {
			result = append(result, reference)
		}
	}

	return result
}

// https://platform.openai.com/docs/api-reference/chat/create?lang=curl
func GenerateCompletion(text string) string {
	completionRequest := NewCompletionRequest()
	completionRequest.AddMessage("system", openAICompletionSystemPrompt)
	completionRequest.AddMessage("user", text)

	completionResponse := NewCompletionResponse()

	fromJSON(doPost(openAICompletionURL, toJSON(completionRequest)), completionResponse)

	return completionResponse.Content()
}

// https://platform.openai.com/docs/api-reference/embeddings?lang=curl
func GenerateEmbedding(text string) []float64 {
	embeddingRequest := NewEmbeddingRequest(text)
	embeddingResponse := NewEmbeddingResponse()

	fromJSON(doPost(openAIEmbeddingURL, toJSON(embeddingRequest)), embeddingResponse)

	return embeddingResponse.Embedding()
}

func ExpandQuery(query string) QueryExpansionResponse {
	response := GenerateCompletion(string(RenderTemplate("expand", map[string]interface{}{
		"Query": query,
	})))

	expandedQueryResponse := QueryExpansionResponse{}

	fromJSON([]byte(response), &expandedQueryResponse)

	return expandedQueryResponse
}

func GenerateCompletionWithCitations(query string, references []Reference) CitedResponse {
	response := GenerateCompletion(string(RenderTemplate("response", map[string]interface{}{
		"Query":      query,
		"References": references,
	})))

	citedResponse := CitedResponse{}

	fromJSON([]byte(response), &citedResponse)

	return citedResponse
}

func Handler(w http.ResponseWriter, r *http.Request) {
	title := "Study"

	switch r.Method {
	case http.MethodPost:
		query := r.FormValue("query")

		var references []Reference
		for _, query := range ExpandQuery(query).Queries {
			references = append(references, FindReference(query)...)
		}

		w.Write(RenderTemplate("index", map[string]interface{}{
			"Title":    title,
			"Response": GenerateCompletionWithCitations(query, references),
		}))
	case http.MethodGet:
		w.Write(RenderTemplate("index", map[string]interface{}{
			"Title":    title,
			"Response": nil,
		}))
	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}

func main() {
	log.SetFlags(log.LstdFlags | log.Lmicroseconds | log.Lmsgprefix)
	log.Default().SetPrefix("study ")

	openAIAPIKey = mustGetEnv(openAIAPIKeyEnv)
	openAILanguageModel = getEnv(openAILanguageModelEnv, openAILanguageModelDefault)
	openAIEmbeddingModel = getEnv(openAIEmbeddingModelEnv, openAIEmbeddingModelDefault)

	LoadTemplate("index", TMPL_INDEX)
	LoadTemplate("expand", TMPL_PROMPT_EXPAND_QUERY)
	LoadTemplate("response", TMPL_PROMPT_RESPONSE)

	LoadReferences(referencesPath)

	http.HandleFunc("/", Handler)
	http.Handle("/references/", http.StripPrefix("/references/", http.FileServer(http.Dir("references"))))
	http.ListenAndServe("127.0.0.1:8080", nil)
}
