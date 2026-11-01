-- Put this Script in ServerScriptService.
-- Enable: Game Settings > Security > Allow HTTP Requests.
-- Change the URL to your deployed server.
local HttpService = game:GetService("HttpService")

local API_URL = "https://YOUR-SERVER.example.com"

-- Example function: pixels must be a Lua array of exactly 324 numbers (0 to 1).
local function recognizeDigit(pixels)
    assert(#pixels == 324, "Expected 324 pixel values for an 18x18 image")

    local body = HttpService:JSONEncode({pixels = pixels})
    local response = HttpService:RequestAsync({
        Url = API_URL .. "/predict",
        Method = "POST",
        Headers = {["Content-Type"] = "application/json"},
        Body = body
    })

    if not response.Success then
        error("Digit server returned HTTP " .. response.StatusCode .. ": " .. response.Body)
    end

    return HttpService:JSONDecode(response.Body)
end

-- Demo with a blank image. Replace this with pixels collected from your drawing GUI.
local blankPixels = table.create(324, 0)
local result = recognizeDigit(blankPixels)
print("Recognized digit:", result.digit)
print("Confidence:", result.confidence)
