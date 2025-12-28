"""
KITE CONNECT TOKEN GENERATOR
Run this script daily to generate your access token
"""

import os
import sys

try:
    from kiteconnect import KiteConnect
except ImportError:
    print("❌ KiteConnect not installed!")
    print("Install it with: pip install kiteconnect")
    sys.exit(1)

def generate_token():
    """Generate Kite Connect access token"""
    
    print("\n" + "="*60)
    print("🔑 KITE CONNECT TOKEN GENERATOR")
    print("="*60 + "\n")
    
    # Get API credentials
    print("📋 Enter your Kite Connect credentials:")
    print("(Get these from: https://developers.kite.trade/)\n")
    
    api_key = input("API Key: ").strip()
    if not api_key:
        print("❌ API Key is required!")
        return
    
    api_secret = input("API Secret: ").strip()
    if not api_secret:
        print("❌ API Secret is required!")
        return
    
    # Initialize Kite
    kite = KiteConnect(api_key=api_key)
    
    # Generate login URL
    login_url = kite.login_url()
    
    print("\n" + "="*60)
    print("STEP 1: LOGIN TO ZERODHA")
    print("="*60)
    print("\n📱 Visit this URL in your browser:\n")
    print(f"🔗 {login_url}\n")
    print("1. Login with your Zerodha credentials")
    print("2. Authorize the app")
    print("3. You'll be redirected to a URL like:")
    print("   http://127.0.0.1/?request_token=XXXXX&action=login&status=success")
    print("4. Copy the 'request_token' from that URL")
    print("="*60 + "\n")
    
    # Get request token
    request_token = input("Enter the request_token: ").strip()
    
    if not request_token:
        print("❌ Request token is required!")
        return
    
    # Generate session
    try:
        print("\n⏳ Generating access token...")
        
        data = kite.generate_session(request_token, api_secret=api_secret)
        access_token = data["access_token"]
        user_id = data["user_id"]
        
        print("\n" + "="*60)
        print("✅ SUCCESS! ACCESS TOKEN GENERATED")
        print("="*60)
        print(f"\n👤 User ID: {user_id}")
        print(f"🔑 API Key: {api_key}")
        print(f"🎫 Access Token:\n{access_token}")
        print("\n⏰ Valid until: 6:00 AM tomorrow (IST)")
        print("="*60 + "\n")
        
        # Save to file
        save_option = input("💾 Save credentials to file? (y/n): ").strip().lower()
        
        if save_option == 'y':
            # Save to .streamlit/secrets.toml
            save_to_streamlit = input("Save to .streamlit/secrets.toml? (y/n): ").strip().lower()
            
            if save_to_streamlit == 'y':
                os.makedirs('.streamlit', exist_ok=True)
                
                with open('.streamlit/secrets.toml', 'w') as f:
                    f.write(f'# Kite Connect Credentials\n')
                    f.write(f'# Generated: {data.get("login_time", "")}\n\n')
                    f.write(f'KITE_API_KEY = "{api_key}"\n')
                    f.write(f'KITE_ACCESS_TOKEN = "{access_token}"\n')
                
                print("\n✅ Saved to .streamlit/secrets.toml")
            
            # Save to .env
            save_to_env = input("Save to .env file? (y/n): ").strip().lower()
            
            if save_to_env == 'y':
                with open('.env', 'w') as f:
                    f.write(f'# Kite Connect Credentials\n')
                    f.write(f'KITE_API_KEY={api_key}\n')
                    f.write(f'KITE_ACCESS_TOKEN={access_token}\n')
                
                print("✅ Saved to .env file")
            
            # Save token only
            with open('.kite_token', 'w') as f:
                f.write(access_token)
            
            print("✅ Token saved to .kite_token file")
        
        print("\n" + "="*60)
        print("🚀 NEXT STEPS:")
        print("="*60)
        print("1. Run the trading bot:")
        print("   streamlit run institutional_algo_trader.py")
        print("\n2. The bot will automatically use these credentials")
        print("\n3. Remember to regenerate token tomorrow!")
        print("="*60 + "\n")
        
        # Test connection
        test = input("🧪 Test connection now? (y/n): ").strip().lower()
        
        if test == 'y':
            test_connection(api_key, access_token)
    
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\n🔍 Common issues:")
        print("1. Wrong API secret")
        print("2. Invalid request token")
        print("3. Token already used (generate new one)")
        print("4. App not authorized on Kite Console")

def test_connection(api_key, access_token):
    """Test the connection"""
    try:
        print("\n⏳ Testing connection...")
        
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)
        
        # Get profile
        profile = kite.profile()
        
        print("\n" + "="*60)
        print("✅ CONNECTION TEST SUCCESSFUL!")
        print("="*60)
        print(f"Name: {profile['user_name']}")
        print(f"Email: {profile['email']}")
        print(f"Broker: {profile['broker']}")
        print("="*60)
        
        # Test market data
        print("\n📊 Testing market data...")
        ltp = kite.ltp(["NSE:RELIANCE", "NSE:TCS", "NSE:INFY"])
        
        print("\nLive Prices:")
        for instrument, data in ltp.items():
            symbol = instrument.split(':')[1]
            price = data['last_price']
            print(f"  {symbol}: ₹{price:,.2f}")
        
        # Test margins
        print("\n💰 Testing account info...")
        margins = kite.margins()
        equity = margins.get('equity', {})
        available = equity.get('available', {}).get('cash', 0)
        
        print(f"\nAvailable Margin: ₹{available:,.2f}")
        
        print("\n✅ All systems operational!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Connection test failed: {e}")

def main():
    """Main function"""
    try:
        generate_token()
    except KeyboardInterrupt:
        print("\n\n⚠️ Cancelled by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
