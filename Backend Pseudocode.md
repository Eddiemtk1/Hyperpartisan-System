INITIALISE FastAPI
CONFIGURE CORS to allow reuests from the extension
INITIALISE OpenRouter API

DEFINE ENDPOINT POST '/analyse':
    RECIEVE article text and title from the request body

    IF article text is empty:
        RETURN HTTP 400 Error 'no text providded'

    DEFINE system prompt with hyperpartisan rules and examples

    TRY:
        SEND prompt and article text to LLM
        REIEVE JSON response from LLM

        EXTRACT results from response

        IF sentence is hyperpartisan AND overall confidence < 0.50:
            SET sentece to false hyperpartisan

        FOR EACH item in results:
            FIND the closest matching sentence in the original article text
            IF match found:
                UPDATE item;s sentence to the exact match from the orignal text

        REMOVE duplicates from results
        SORT results by their chronological appearance in the article text
        KEEP only the top 5 results

        RETURN JSON object containing (is hyperpartisan, overfall confidence, biased results)
        
    CATCH errors:
        RETURN HTTP 500 error