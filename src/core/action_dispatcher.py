"""
Action Dispatcher untuk Mochi Bot
Mengirim alert ke Mobile App
"""

import json
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class AlertPayload:
    """Payload untuk alert ke mobile app"""
    alert_type: str
    priority: str
    function:  str
    arguments: Dict[str, Any]
    user_id: str
    timestamp:  str


class ActionDispatcher:
    """Dispatcher untuk mengirim action ke mobile app"""
    
    def __init__(
        self,
        api_endpoint: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        
        # Alert type mapping
        self.alert_mapping = {
            "alert_caregiver": {"type": "EMERGENCY", "priority": "CRITICAL", "icon": "🚨"},
            "request_service": {"type": "SERVICE", "priority": "NORMAL", "icon": "☕"},
            "request_assistance": {"type": "ASSISTANCE", "priority": "HIGH", "icon": "🤝"}
        }
    
    def dispatch(
        self,
        function_call: Dict[str, Any],
        user_id: str = "user_001"
    ) -> AlertPayload:
        """Dispatch action ke mobile app"""
        
        func_name = function_call["function"]
        func_args = function_call["arguments"]
        
        alert_info = self.alert_mapping.get(
            func_name,
            {"type": "UNKNOWN", "priority": "NORMAL", "icon": "❓"}
        )
        
        payload = AlertPayload(
            alert_type=alert_info["type"],
            priority=alert_info["priority"],
            function=func_name,
            arguments=func_args,
            user_id=user_id,
            timestamp=datetime.now().isoformat()
        )
        
        # Print untuk debugging/simulasi
        self._print_alert(payload, alert_info["icon"])
        
        # Jika ada API endpoint, kirim request
        if self.api_endpoint:
            self._send_to_api(payload)
        
        return payload
    
    def _print_alert(self, payload: AlertPayload, icon: str):
        """Print alert ke console"""
        print("\n" + "=" * 55)
        print(f"{icon} ALERT TERKIRIM KE APLIKASI MOBILE {icon}")
        print("=" * 55)
        print(f"  📱 Tipe Alert  : {payload.alert_type}")
        print(f"  ⚡ Prioritas   : {payload.priority}")
        print(f"  👤 User ID     : {payload.user_id}")
        print(f"  📦 Function    : {payload.function}")
        print(f"  📄 Arguments   : {json.dumps(payload.arguments, ensure_ascii=False)}")
        print(f"  🕐 Timestamp   : {payload. timestamp}")
        print("=" * 55)
    
    def _send_to_api(self, payload: AlertPayload):
        """Kirim payload ke API endpoint (implementasi nanti)"""
        # TODO: Implementasi HTTP POST ke mobile app backend
        pass