class InteractionAgent:
    def __init__(self, database):
        self.db = database # Dict: {'book_name': 'Author/Status'}
        self.intents = {
            'find': ['where', 'search', 'find', 'looking'],
            'status': ['available', 'checked', 'status'],
            'greet': ['hi', 'hello', 'hey']
        }

    def process_query(self, user_input):
        user_input = user_input.lower()
        
        # 1. Detect Intent
        detected_intent = None
        for intent, keywords in self.intents.items():
            if any(k in user_input for k in keywords):
                detected_intent = intent
                break
        
        # 2. Extract Entities (Basic)
        target_item = None
        for item in self.db.keys():
            if item.lower() in user_input:
                target_item = item
                break

        # 3. Execute Response
        if detected_intent == 'find' and target_item:
            return f"The book '{target_item}' is located at {self.db[target_item]['location']}."
        elif detected_intent == 'status' and target_item:
            return f"Status: {self.db[target_item]['status']}"
        elif detected_intent == 'greet':
            return "Hello! How can I help you with the library today?"
        else:
            return "I didn't understand. Try asking about a book status."