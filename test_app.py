import sys
import time
import requests

# Set stdout to UTF-8 to prevent Windows console UnicodeEncodeError
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

SERVER_URL = "http://127.0.0.1:5000/predict"

TEST_CASES = [
    {
        "id": 1,
        "type": "Spam",
        "email": "CONGRATULATIONS! You won $1,000,000 lottery jackpot! Click http://claim-lottery.com to claim your reward instantly!",
        "expected": "Spam"
    },
    {
        "id": 2,
        "type": "Spam",
        "email": "URGENT: Your bank account access has been restricted due to unauthorized login! Verify SSN and bank details immediately.",
        "expected": "Spam"
    },
    {
        "id": 3,
        "type": "Spam",
        "email": "Make $5,000 a week working from home! Guaranteed 200% ROI crypto trading bot. Sign up for free demo!",
        "expected": "Spam"
    },
    {
        "id": 4,
        "type": "Ham",
        "email": "Hi Sriram, please find attached the weekly status report for the engineering team. Let me know if you have any questions.",
        "expected": "Ham"
    },
    {
        "id": 5,
        "type": "Ham",
        "email": "Hey, are we still meeting for lunch at 12:30 PM today? Let me know if that place around the corner works for you.",
        "expected": "Ham"
    }
]

def run_tests():
    print("="*65)
    print("      EMAIL SPAM DETECTOR - AUTOMATED API TEST SUITE      ")
    print("="*65)
    print(f"Target Endpoint: {SERVER_URL}\n")
    
    passed = 0
    failed = 0

    for test in TEST_CASES:
        print(f"Test #{test['id']} [{test['type']} Test]")
        print(f"Text Snippet: \"{test['email'][:70]}...\"")
        
        try:
            response = requests.post(SERVER_URL, json={"text": test['email']}, timeout=5)
            
            if response.status_code != 200:
                print(f"  [FAIL] HTTP Status Code {response.status_code}")
                failed += 1
                continue
                
            data = response.json()
            prediction = data.get("prediction")
            confidence = data.get("confidence", 0.0)
            
            if prediction == test['expected']:
                print(f"  [PASS] Prediction: {prediction} | Confidence: {confidence * 100:.1f}%")
                passed += 1
            else:
                print(f"  [FAIL] Expected: {test['expected']} | Got: {prediction} | Confidence: {confidence * 100:.1f}%")
                failed += 1
                
        except Exception as e:
            print(f"  [FAIL] Exception occurred - {str(e)}")
            failed += 1
            
        print("-" * 65)

    print("\n" + "="*65)
    print(f"TEST SUMMARY: {passed}/{len(TEST_CASES)} Passed, {failed} Failed.")
    print("="*65)

    if failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    time.sleep(1)
    run_tests()
