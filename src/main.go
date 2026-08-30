package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"
)

type Evaluations struct {
	IntelIndex  *float64 `json:"artificial_analysis_intelligence_index"`
	CodingIndex *float64 `json:"artificial_analysis_coding_index"`
	MathIndex   *float64 `json:"artificial_analysis_math_index"`
	MMLUPro     *float64 `json:"mmlu_pro"`
	GPQA        *float64 `json:"gpqa"`
	HLE         *float64 `json:"hle"`
	LCB         *float64 `json:"livecodebench"`
	SciCode     *float64 `json:"scicode"`
	Math500     *float64 `json:"math_500"`
	AIME        *float64 `json:"aime"`
	AIME25      *float64 `json:"aime_25"`
	IFBench     *float64 `json:"ifbench"`
	LCR         *float64 `json:"lcr"`
	TBHard      *float64 `json:"terminalbench_hard"`
	Tau2        *float64 `json:"tau2"`
}

type ModelCreator struct {
	ID   string `json:"id"`
	Name string `json:"name"`
	Slug string `json:"slug"`
}

type ModelData struct {
	ID            string       `json:"id"`
	Name          string       `json:"name"`
	Slug          string       `json:"slug"`
	ReleaseDate   string       `json:"release_date"`
	ModelCreator  ModelCreator `json:"model_creator"`
	Evaluations   *Evaluations `json:"evaluations,omitempty"`
	ParameterSize float64      `json:"parameter_size"`
}

type APIResponse struct {
	Status int         `json:"status"`
	Data   []ModelData `json:"data"`
}

var openWeightOrgs = map[string]bool{
	"qwen":             true,
	"deepseek":         true,
	"zai-org":          true,
	"moonshotai":       true,
	"meta":             true,
	"mistral":          true,
	"mistralai":        true,
	"01-ai":            true,
	"baichuan":         true,
	"cohere":           true,
	"falcon":           true,
	"tii":              true,
	"tii-uae":          true,
	"teknium":          true,
	"togethercomputer": true,
	"mosaicml":         true,
	"allenai":          true,
	"eleutherai":       true,
	"bigscience":       true,
	"databricks":       true,
	"tiiuae":           true,
	"internlm":         true,
	"baai":             true,
	"openchat":         true,
	"phi":              true,
	"microsoft":        true,
	"wizardlm":         true,
	"lmsys":            true,
	"stability":        true,
	"stabilityai":      true,
	"nexusflow":        true,
	"upstage":          true,
	"alibaba":          true,
	"azure":            true,
	"ai2":              true,
	"nous-research":    true,
	"nvidia":           true,
}

var openWeightPatterns = []string{
	"llama",
	"mistral",
	"mixtral",
	"deepseek",
	"qwen",
	"phi-",
	"falcon",
	"gemma",
	"gpt-oss",
	"glm-",
	"kimi-k2",
	"yi-",
	"internlm",
	"baichuan",
	"command-r",
	"aya",
	"dbrx",
	"olmo",
	"solar",
	"molmo",
}

var closedOrgs = map[string]bool{
	"openai":     true,
	"anthropic":  true,
	"google":     true,
	"xai":        true,
	"cohere-api": true,
}

var (
	sizeRegex     = regexp.MustCompile(`(?i)(\d+(?:\.\d+)?)\s*([bm])\b`)
	eSizeRegex    = regexp.MustCompile(`(?i)e(\d+(?:\.\d+)?)\s*b\b`)
	oldLlamaRegex = regexp.MustCompile(`(?i)llama\s*[1-3]`)
)

var knownSizesFallback = map[string]float64{
	"llama-4-maverick": 14.0,
	"phi-4-mini":       4.0,
	"phi-4-multimodal": 6.0,
}

func isOpenWeights(model ModelData) bool {
	nameLower := strings.ToLower(model.Name)
	slugLower := strings.ToLower(model.Slug)
	creatorLower := strings.ToLower(model.ModelCreator.Slug)

	if oldLlamaRegex.MatchString(nameLower) || oldLlamaRegex.MatchString(slugLower) {
		return false
	}

	if strings.Contains(nameLower, "max") && (strings.Contains(nameLower, "qwen") || strings.Contains(slugLower, "qwen") || creatorLower == "alibaba") {
		return false
	}

	for _, pat := range openWeightPatterns {
		if strings.Contains(nameLower, pat) || strings.Contains(slugLower, pat) {
			return true
		}
	}

	if closedOrgs[creatorLower] {
		return false
	}

	if openWeightOrgs[creatorLower] {
		return true
	}

	return false
}

func getParameterSize(model ModelData) (float64, bool) {
	nameLower := strings.ToLower(model.Name)
	slugLower := strings.ToLower(model.Slug)

	for _, text := range []string{nameLower, slugLower} {
		if matches := eSizeRegex.FindStringSubmatch(text); len(matches) > 1 {
			if val, err := strconv.ParseFloat(matches[1], 64); err == nil {
				return val, true
			}
		}

		if matches := sizeRegex.FindStringSubmatch(text); len(matches) > 2 {
			val, err := strconv.ParseFloat(matches[1], 64)
			if err == nil {
				unit := matches[2]
				if unit == "m" {
					return val / 1000.0, true
				}
				return val, true
			}
		}
	}

	if val, ok := knownSizesFallback[slugLower]; ok {
		return val, true
	}

	return 0, false
}

func fetchModelsWithRetry(apiKey string) ([]ModelData, error) {
	url := "https://artificialanalysis.ai/api/v2/data/llms/models"
	maxRetries := 3
	initialBackoff := 2 * time.Second

	client := &http.Client{Timeout: 15 * time.Second}
	var lastErr error

	for attempt := 1; attempt <= maxRetries; attempt++ {
		req, err := http.NewRequest("GET", url, nil)
		if err != nil {
			return nil, fmt.Errorf("failed to create request: %w", err)
		}

		req.Header.Set("x-api-key", apiKey)
		req.Header.Set("Content-Type", "application/json")

		resp, err := client.Do(req)
		if err != nil {
			lastErr = err
			log.Printf("[Attempt %d/%d] Error: %v. Retrying...", attempt, maxRetries, err)
			time.Sleep(initialBackoff)
			initialBackoff *= 2
			continue
		}

		if resp.StatusCode != http.StatusOK {
			lastErr = fmt.Errorf("unexpected status: %d", resp.StatusCode)
			rbcErr := resp.Body.Close()
			if rbcErr != nil {
				log.Printf("failed to close response body: %v", rbcErr)
			}
			time.Sleep(initialBackoff)
			initialBackoff *= 2
			continue
		}

		var apiResp APIResponse
		decodeErr := json.NewDecoder(resp.Body).Decode(&apiResp)
		rbcErr := resp.Body.Close()
		if rbcErr != nil {
			log.Printf("failed to close response body: %v", rbcErr)
		}

		if decodeErr != nil {
			return nil, fmt.Errorf("failed to decode JSON: %w", decodeErr)
		}

		return apiResp.Data, nil
	}

	return nil, fmt.Errorf("all retries failed: %w", lastErr)
}

func saveJSONToFile(filename string, models []ModelData) error {
	data, err := json.MarshalIndent(models, "", "  ")
	if err != nil {
		return fmt.Errorf("JSON encoding error: %w", err)
	}
	err = os.WriteFile(filename, data, 0644)
	if err != nil {
		return fmt.Errorf("file write error: %w", err)
	}

	return nil
}

func resolvePath(path string) (string, error) {
	if strings.HasPrefix(path, "~") {
		home, err := os.UserHomeDir()
		if err != nil {
			return "", fmt.Errorf("could not get home directory: %w", err)
		}
		if path == "~" {
			path = home
		} else if strings.HasPrefix(path, "~/") || strings.HasPrefix(path, "~\\") {
			path = filepath.Join(home, path[2:])
		}
	}
	ext := filepath.Ext(path)
	if strings.ToLower(ext) != ".json" {
		path = path + ".json"
	}
	absPath, err := filepath.Abs(path)
	if err != nil {
		return "", fmt.Errorf("could not resolve absolute path: %w", err)
	}

	dir := filepath.Dir(absPath)
	err = os.MkdirAll(dir, 0755)
	if err != nil {
		log.Fatalf("failed to create directories: %v", err)
	}

	return absPath, nil
}

func main() {
	apiKey := os.Getenv("AA_API_KEY")

	if apiKey == "" {
		log.Fatal("env 'AA_API_KEY' not set!")
	}

	outputFlag := flag.String("output", "filtered_models.json", "path to save the filtered models JSON file")
	flag.Parse()

	models, err := fetchModelsWithRetry(apiKey)
	if err != nil {
		log.Fatalf("Error: %v", err)
	}

	var filteredModels []ModelData
	for _, model := range models {
		if !isOpenWeights(model) {
			continue
		}

		size, found := getParameterSize(model)
		if !found {
			continue
		}

		if size <= 40.0 {
			model.ParameterSize = size
			filteredModels = append(filteredModels, model)
		}
	}

	outputFile, err := resolvePath(*outputFlag)
	if err != nil {
		log.Fatalf("failed to resolve output path: %v", err)
	}

	err = saveJSONToFile(outputFile, filteredModels)
	if err != nil {
		log.Fatalf("failed to save file: %v", err)
	}

	fmt.Printf("Successfully saved models: %d to file: %s\n", len(filteredModels), outputFile)
}
