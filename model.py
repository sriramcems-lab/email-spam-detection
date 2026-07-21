import os
import re
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

DATASET_PATH = "dataset.csv"
MODEL_PATH = "spam_model.joblib"
VECTORIZER_PATH = "tfidf_vectorizer.joblib"

# Preprocessing function
def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove HTML tags if present
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove URL links
    text = re.sub(r'http[s]?://\S+|www\.\S+', ' ', text)
    # Remove special characters and punctuation
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def generate_mock_dataset():
    """Generates a rich, realistic 120-sample dataset for Email Spam/Ham detection."""
    spam_samples = [
        "URGENT: Your bank account has been suspended! Click here to verify your credentials immediately: http://bit.ly/bank-security-fix",
        "Congratulations! You have been selected to receive a free $1,000 Amazon Gift Card! Claim your reward now by clicking this link.",
        "Exclusive offer: Make $5,000 per week working from home! No experience required. Guaranteed high return on investment.",
        "Dear Winner, your email address won $2,500,000 in the International Lottery 2026. Contact agent to claim your jackpot payout.",
        "FINAL NOTICE: Tax refund of $480 is pending approval. Confirm your routing number immediately at http://secure-tax-refund.com",
        "Get cheap Rx meds online without prescription! Pfizer Viagr4, Cial1s, Amoxil available at 80% discount. Order today!",
        "Double your Bitcoins in 24 hours! Automated crypto trading bot with guaranteed 200% ROI. Join our telegram channel now.",
        "You have 1 unread secure message from Wells Fargo Security Team. Log in now to stop fraudulent transaction of $1,450.",
        "Special Discount! 90% OFF designer watches, Rolex, Omega, Gucci replicas. Free worldwide shipping on all orders!",
        "Urgent assistance needed: I am Barrister James Campbell, handling $10.5M inheritance. Reply with your full name and bank info.",
        "Act Now! Pre-approved credit card with $50,000 limit and 0% APR. No credit check required! Apply instantly.",
        "ALERT: Someone attempted to login to your Apple ID from Moscow, Russia. If this wasn't you, reset password immediately.",
        "Lose 20 lbs in 14 days with this miracle keto pill! Doctors don't want you to know this secret weight loss formula.",
        "Hot singles in your area want to meet tonight! Click to view confidential profiles and chat live.",
        "Refinance your mortgage at record low interest rates! Save up to $700 every month. Get free quote today.",
        "Claim your complimentary iPhone 16 Pro Max! Only 3 left in stock for selected lucky users. Complete survey now.",
        "IMPORTANT: Netflix subscription suspended due to failed billing. Update payment details within 24 hours to avoid cancellation.",
        "Instant cash loans up to $5,000 deposited into your checking account within 1 hour! Bad credit accepted.",
        "Warning: Your cloud storage is 99% full! Upgrade to Premium 2TB now for only $1.99 or risk losing your photos.",
        "Earn passive income with AI arbitrage trading! Zero risk, high reward system. Sign up for free demo account.",
        "RE: Your unclaimed inheritance from late uncle. Transfer fee required to release $3,400,000 USD via wire transfer.",
        "Free casino spins! Deposit $10 and get $500 bonus cash instantly. Cash out winnings immediately without restrictions.",
        "Your PayPal account access has been restricted due to unauthorized login attempts. Click here to verify identity.",
        "Work 2 hours a day and earn $800 daily! Simple data entry job online. Immediate start available.",
        "Exclusive clearance sale: Ray-Ban sunglasses starting at $19.99! Limited stock available, buy now before sold out.",
        "You have won a brand new Tesla Model Y! Click the verification link to claim your vehicle before timer runs out.",
        "URGENT: Social Security Number compromised! Call our legal department immediately at 1-800-FAKE-NUM to prevent arrest warrant.",
        "Boost your website traffic by 10,000 daily visitors! Affordable SEO package starting at $29/month.",
        "Get verified checkmark on Instagram & X instantly! Guaranteed verification service for influencers and businesses.",
        "Don't miss out! Limited time opportunity to invest in top pre-IPO tech startup. Expected 10x returns.",
        "Security Alert: Microsoft Account breached! Click link to lock account and scan for trojans.",
        "Cheap flight deals! Fly to Paris, London, Tokyo for under $150 roundtrip. Book before seats sell out.",
        "You have received a $500 Walmart E-Gift card. Click here to claim your reward before midnight.",
        "WARNING: Malware detected on your device! Download our antivirus shield immediately to protect your files.",
        "Get a university degree online in 2 weeks! No studying or exams required. Accredited diplomas delivered to your doorstep.",
        "Your package delivery failed! Update your shipping address and pay $1.99 redelivery fee at http://usps-tracking-update.com",
        "Make $300 an hour testing mobile apps from home! No experience required. Start earning today.",
        "CONGRATULATIONS: You are today's visitor #100,000! Spin the wheel to win cash prizes up to $10,000.",
        "Urgent: Your domain name is expiring today! Renew immediately to prevent domain deletion and website shutdown.",
        "Cheap prescription drugs online! Buy generic painkillers, sleep aids, and antibiotics delivered discreetly.",
        "Your Amazon account is locked due to suspicious activity. Click here to verify your credit card details.",
        "Make money with online poker! $1,000 welcome bonus for new players. Play live tables and win real cash.",
        "Low rate personal loans guaranteed approval! Money in your account within 24 hours. Apply now.",
        "Alert: Unusual activity on your Visa card. $899.00 charge at Target. Click here to dispute charge.",
        "Free trial of premium anti-aging serum! Look 10 years younger in just 7 days. Pay only shipping fee.",
        "RE: Payment overdue for Invoice #9842. Pay $3,450 immediately to avoid legal collections action.",
        "Get instant followers and likes on TikTok! Guaranteed viral growth for your account.",
        "Your wire transfer of $12,500 is currently pending approval. Click to cancel or confirm transaction.",
        "Urgent message regarding your pension plan. You qualify for an additional lump sum payout of $45,000.",
        "Special invitation: Join private wealth management group with guaranteed 35% annual returns.",
        "Your Google Drive quota exceeded! Clear space or purchase storage plan to continue receiving emails.",
        "Free $250 gas card for essential workers! Complete 3 minute questionnaire to receive yours in mail.",
        "WARNING: Your Windows license key has expired! Renew now to prevent system shutdown.",
        "Exclusive opportunity to purchase wholesale electronics: iPhones, iPads, MacBooks at 70% below retail.",
        "Urgent: Action required on your tax return. Claim your additional stimulus payment now.",
        "Get 500 free spins on mega slots! No deposit required. Win real cash jackpots today.",
        "Your eBay order #402-99823 has been shipped. Click to track package or cancel unauthorized order.",
        "Financial aid notification: You are eligible for $7,500 federal education grant. Claim your grant today.",
        "Alert: Password change requested for your Coinbase account. If you did not request this, click here immediately.",
        "Earn up to $50/hour filling out paid market research surveys from your phone! Sign up free today."
    ]

    ham_samples = [
        "Hi Team, attached is the revised project roadmap for Q3. Please review and let me know your thoughts before tomorrow's meeting.",
        "Hey John, are we still on for lunch today at 12:30 PM? Let me know if that place around the corner works for you.",
        "Dear Sriram, thank you for submitting your application for the Senior Software Engineer position. We would like to schedule an initial interview.",
        "Here is the weekly status report for the engineering department. All milestones were achieved on schedule.",
        "Reminder: The quarterly all-hands meeting will take place on Thursday at 10:00 AM in the main auditorium.",
        "Hi Mom, just wanted to check in and see how you are doing. Let's catch up over the phone this weekend!",
        "Your flight reservation for flight AA-1420 to San Francisco is confirmed. Confirmation code: G7X9K2.",
        "Please find attached the receipt for your recent order #88492 at the Bookstore. Thank you for your purchase!",
        "Hi David, could you please review my pull request on GitHub when you have a chance? I updated the unit tests.",
        "Your monthly electric bill of $78.45 is now available online. Payment will be automatically deducted on July 25th.",
        "Good morning team, please note that office maintenance will be conducted this Saturday between 9 AM and 2 PM.",
        "Hey, thanks for sharing that article on machine learning model optimization! It was really helpful for our architecture.",
        "Hi Sarah, can you send over the updated slide deck for the client presentation tomorrow morning?",
        "Your appointment with Dr. Smith is confirmed for Tuesday, July 22 at 3:00 PM. Please arrive 15 minutes early.",
        "Hi everyone, the code freeze for the upcoming release v2.4 will start on Friday at midnight.",
        "Thanks for the quick reply! I'll go ahead and finalize the vendor contract based on these terms.",
        "Your subscription invoice for Spotify Family Plan ($16.99) is ready. View details in your account dashboard.",
        "Hi team, just a quick heads up that the staging environment will undergo scheduled maintenance tonight at 11 PM.",
        "Hey Alex, do you have a few minutes for a quick huddle on Zoom to discuss the database migration strategy?",
        "Hi all, please welcome our new designer Maya to the team! She brings extensive UI/UX design experience.",
        "Your library books are due in 3 days. Log in to renew your loans or return them to avoid late fees.",
        "Hey, don't forget to submit your timesheet for this pay period by 5 PM today.",
        "Thanks for joining our webinar on Cloud Architecture Best Practices. You can download the slides and recording here.",
        "Hi Sriram, here are the meeting notes from today's discussion on API design and data models.",
        "Hi team, the customer support queue has been cleared for today. Great job everyone!",
        "Hi, your table reservation at Bistro Italia for 4 people on Friday at 7:30 PM is confirmed.",
        "Hi standard shipping update: Your package from Amazon has been delivered to your front door.",
        "Hey, do you know where the documentation for the legacy authentication service is stored?",
        "Hi, please find the signed nondisclosure agreement attached for your records.",
        "Reminder: Team standup is moving to 9:30 AM starting next week to accommodate remote team members in different timezones.",
        "Hi Professor, I have a question regarding question 3 on the homework assignment due this Thursday.",
        "Your hotel booking at Marriott Downtown for 2 nights starting August 10th is confirmed.",
        "Hi all, the office HVAC system has been repaired and temperature control is back to normal.",
        "Thanks for organizing the team dinner last night! It was great catching up with everyone outside of work.",
        "Hi team, please update your JIRA tickets with the latest progress before our sprint retrospective.",
        "Your monthly internet service bill of $65.00 is ready for viewing. Thank you for being a valued customer.",
        "Hi Mark, I've updated the design specifications document with your feedback on the mobile layout.",
        "Hey, are you free this weekend to go hiking at the national park? Weather forecast looks great!",
        "Hi all, please remember to lock your workstations when leaving your desk for security compliance.",
        "Your dentist appointment is scheduled for next Monday at 10:00 AM. Reply C to confirm.",
        "Hi Sriram, thanks for submitting the code review feedback. I've addressed all comments in commit a8f92b.",
        "Good afternoon, attached is the monthly newsletter with updates on company events and accomplishments.",
        "Hi team, the release candidate build for v3.0 has passed all regression tests successfully.",
        "Hey, do you want to grab coffee from the cafeteria before the 2 PM product review meeting?",
        "Hi Rachel, can you double-check the budget figures in column C of the financial report?",
        "Your car service appointment at Honda Service Center is confirmed for Thursday at 8:00 AM.",
        "Hi team, we have added new design system components to Figma. Check the design channel for details.",
        "Hi Sriram, here is the draft blog post for our upcoming product feature announcement. Let me know your feedback.",
        "Hey, just following up on our email from yesterday regarding the API rate limiting implementation.",
        "Hi everyone, the company picnic is scheduled for Saturday, August 15th at Central Park. RSVP by Friday!",
        "Your dry cleaning is ready for pickup at Main Street Cleaners.",
        "Hi Alex, attached is the revised proposal with the updated pricing table we discussed.",
        "Hi team, the server monitoring alert has been resolved. CPU usage is back to normal parameters.",
        "Hey, thanks for helping out with the onboarding setup for the new team members today!",
        "Hi, your rental car reservation with Hertz for San Jose Airport is confirmed.",
        "Hi Sriram, please review the revised SLA agreement document when you have a moment.",
        "Hi everyone, please remember to submit your annual benefit selection preferences by the end of the month.",
        "Hey, let me know if you need any assistance with the backend deployment pipeline.",
        "Hi team, great work on hitting our Q2 performance targets ahead of schedule!",
        "Hi Sriram, here are the slides from the tech talk on microservices architecture."
    ]

    df_spam = pd.DataFrame({'text': spam_samples, 'label': 'Spam'})
    df_ham = pd.DataFrame({'text': ham_samples, 'label': 'Ham'})
    df = pd.concat([df_spam, df_ham], ignore_index=True)
    
    # Shuffle dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(DATASET_PATH, index=False)
    print(f"[Dataset] Created {len(df)} samples ({len(df_spam)} Spam, {len(df_ham)} Ham) -> saved to '{DATASET_PATH}'")
    return df

def train_and_save_model():
    if not os.path.exists(DATASET_PATH):
        df = generate_mock_dataset()
    else:
        df = pd.read_csv(DATASET_PATH)
        print(f"[Dataset] Loaded {len(df)} samples from '{DATASET_PATH}'")

    # Clean text
    df['clean_text'] = df['text'].apply(preprocess_text)

    X = df['clean_text']
    y = df['label']

    # 80/20 train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        max_features=5000,
        min_df=1
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Classifier (Multinomial Naive Bayes)
    classifier = MultinomialNB(alpha=0.1)
    classifier.fit(X_train_vec, y_train)

    # Evaluate
    y_pred = classifier.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*50)
    print("      MODEL TRAINING & EVALUATION REPORT      ")
    print("="*50)
    print(f"Accuracy Score: {acc * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("="*50 + "\n")

    # Save model artifacts
    joblib.dump(classifier, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"[Model Saved] Classifier -> '{MODEL_PATH}'")
    print(f"[Vectorizer Saved] Vectorizer -> '{VECTORIZER_PATH}'")

if __name__ == "__main__":
    train_and_save_model()
