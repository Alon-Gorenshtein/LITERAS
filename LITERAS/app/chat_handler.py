import json
from fastapi import WebSocket
from .agents import AcademicSearchTeam

class ChatHandler:
    def __init__(self, openai_api_key: str):
        self.search_team = AcademicSearchTeam(api_key=openai_api_key)
    
    def serialize_object(self, obj):
        """
        Custom serialization method to handle complex objects
        """
        try:
            # Handle specific known non-serializable types
            if hasattr(obj, '__class__'):
                # If it's a custom object with a known structure
                if obj.__class__.__name__ == 'FunctionCall':
                    return {
                        "id": getattr(obj, 'id', None),
                        "function": getattr(obj, 'function', None),
                        "arguments": getattr(obj, 'arguments', None)
                    }
                
                # Convert object to dictionary if possible
                if hasattr(obj, '__dict__'):
                    return obj.__dict__
            
            # Handle other special types
            if isinstance(obj, (set, tuple)):
                return list(obj)
            
            # Fallback: convert to string
            return str(obj)
        
        except Exception as e:
            return f"Unserializable object of type {type(obj).__name__}: {e}"
    
    async def handle_websocket(self, websocket: WebSocket):
        await websocket.accept()
        
        try:
            while True:
                message = await websocket.receive_text()
                
                # Process messages with robust serialization
                async for update in self.search_team.process_query(message):
                    # Filter out unnecessary agent updates
                    if update.get('agent') not in ['SearchAgent', 'QueryPlanner', 'Critic', 'Validator','SynthesisAgent','ReferenceConsistencyCritic']:
                        # Use custom serializer with json.dumps
                        serialized_update = json.dumps(update, default=self.serialize_object)
                        await websocket.send_text(serialized_update)
                
        except Exception as e:
            # Serialize error message as well
            error_message = json.dumps({
                "type": "error",
                "message": str(e)
            })
            await websocket.send_text(error_message)

