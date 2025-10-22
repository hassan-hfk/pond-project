"""
Email Notification Handler for Detection Events
Sends email with video thumbnail and feedback buttons when new detection occurs
"""

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.base import MIMEBase
from email import encoders
from pathlib import Path
import yaml
import cv2
from datetime import datetime
import base64
import os

class EmailNotifier:
    """Handles email notifications for detection events"""
    
    def __init__(self, config_path):
        """Initialize email notifier with config"""
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
        
        # Get email configuration
        email_cfg = self.cfg.get('email', {})
        self.enabled = email_cfg.get('enabled', False)
        self.smtp_server = email_cfg.get('smtp_server', 'smtp.gmail.com')
        self.smtp_port = email_cfg.get('smtp_port', 587)
        self.sender_email = email_cfg.get('sender_email', '')
        self.sender_password = email_cfg.get('sender_password', '')
        self.recipient_email = email_cfg.get('recipient_email', '')
        self.app_url = email_cfg.get('app_url', 'http://localhost:5000')
        
        print(f"[EMAIL] Initialized - Enabled: {self.enabled}")
        if self.enabled and not all([self.sender_email, self.sender_password, self.recipient_email]):
            print("[EMAIL] ⚠️  Warning: Email enabled but credentials incomplete")
    
    def send_detection_email(self, event_id, event_data, video_path):
        """
        Send email notification for a detection event
        
        Args:
            event_id: Database ID of the event
            event_data: Dict containing event information
            video_path: Path to the video file
        """
        if not self.enabled:
            print("[EMAIL] Email notifications disabled")
            return False
        
        if not all([self.sender_email, self.sender_password, self.recipient_email]):
            print("[EMAIL] ❌ Missing email credentials")
            return False
        
        try:
            # Extract video thumbnail
            thumbnail_path = self._extract_thumbnail(video_path)
            
            # Create email
            msg = self._create_email_message(event_id, event_data, thumbnail_path)
            
            # Send email
            success = self._send_email(msg)
            
            # Cleanup thumbnail
            if thumbnail_path and os.path.exists(thumbnail_path):
                os.remove(thumbnail_path)
            
            return success
            
        except Exception as e:
            print(f"[EMAIL] ❌ Error sending email: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _extract_thumbnail(self, video_path):
        """Extract first frame from video as thumbnail"""
        try:
            if not os.path.exists(video_path):
                print(f"[EMAIL] ⚠️  Video not found: {video_path}")
                return None
            
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"[EMAIL] ⚠️  Could not open video: {video_path}")
                return None
            
            success, frame = cap.read()
            cap.release()
            
            if not success:
                print(f"[EMAIL] ⚠️  Could not read frame from video")
                return None
            
            # Resize thumbnail
            frame = cv2.resize(frame, (640, 480))
            
            # Save thumbnail temporarily
            thumbnail_dir = Path(video_path).parent / 'thumbnails'
            thumbnail_dir.mkdir(exist_ok=True)
            thumbnail_path = thumbnail_dir / f"thumb_{Path(video_path).stem}.jpg"
            
            cv2.imwrite(str(thumbnail_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            print(f"[EMAIL] Thumbnail created: {thumbnail_path}")
            
            return str(thumbnail_path)
            
        except Exception as e:
            print(f"[EMAIL] Error extracting thumbnail: {e}")
            return None
    
    def _create_email_message(self, event_id, event_data, thumbnail_path):
        """Create email message with HTML content and thumbnail"""
        msg = MIMEMultipart('related')
        
        # Email headers
        event_type = event_data.get('type', 'detection')
        event_class = event_data.get('class', 'unknown')
        timestamp = datetime.fromtimestamp(event_data.get('timestamp', 0)).strftime("%Y-%m-%d %H:%M:%S")
        
        msg['Subject'] = f"🚨 Detection Alert: {event_class.upper()} - {timestamp}"
        msg['From'] = self.sender_email
        msg['To'] = self.recipient_email
        
        # Create HTML content
        html_content = self._create_html_content(event_id, event_data, timestamp)
        
        # Attach HTML
        msg_alternative = MIMEMultipart('alternative')
        msg.attach(msg_alternative)
        
        html_part = MIMEText(html_content, 'html')
        msg_alternative.attach(html_part)
        
        # Attach thumbnail if available
        if thumbnail_path and os.path.exists(thumbnail_path):
            with open(thumbnail_path, 'rb') as f:
                img_data = f.read()
            
            image = MIMEImage(img_data, name=os.path.basename(thumbnail_path))
            image.add_header('Content-ID', '<thumbnail>')
            msg.attach(image)
        
        return msg
    
    def _create_html_content(self, event_id, event_data, timestamp):
        """Create HTML email content with styling and feedback buttons"""
        event_type = event_data.get('type', 'detection')
        event_class = event_data.get('class', 'unknown')
        height = event_data.get('height_m', 'N/A')
        in_roi = event_data.get('in_roi', False)
        
        # Determine alert color based on event type
        if event_type == 'child_in_water':
            alert_color = '#e74c3c'  # Red
            alert_text = '🚨 CRITICAL ALERT'
        elif event_type == 'person_in_roi':
            alert_color = '#f39c12'  # Orange
            alert_text = '⚠️  WARNING'
        else:
            alert_color = '#3498db'  # Blue
            alert_text = 'ℹ️  DETECTION'
        
        # Feedback button URLs
        correct_url = f"{self.app_url}/api/feedback/{event_id}/correct"
        incorrect_url = f"{self.app_url}/api/feedback/{event_id}/incorrect"
        view_url = f"{self.app_url}/?showEventList=true"
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Detection Alert</title>
</head>
<body style="margin: 0; padding: 0; font-family: Arial, sans-serif; background-color: #f5f5f5;">
    <table role="presentation" style="width: 100%; border-collapse: collapse;">
        <tr>
            <td style="padding: 20px 0;">
                <table role="presentation" style="max-width: 600px; margin: 0 auto; background-color: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                    
                    <!-- Header -->
                    <tr>
                        <td style="background: linear-gradient(135deg, {alert_color}, {alert_color}dd); color: white; padding: 30px; text-align: center;">
                            <h1 style="margin: 0; font-size: 28px; font-weight: bold;">{alert_text}</h1>
                            <p style="margin: 10px 0 0 0; font-size: 18px;">Detection System Notification</p>
                        </td>
                    </tr>
                    
                    <!-- Thumbnail -->
                    <tr>
                        <td style="padding: 0;">
                            <img src="cid:thumbnail" alt="Detection Frame" style="width: 100%; height: auto; display: block; max-height: 400px; object-fit: cover;">
                        </td>
                    </tr>
                    
                    <!-- Event Details -->
                    <tr>
                        <td style="padding: 30px;">
                            <h2 style="color: #2c3e50; margin: 0 0 20px 0; font-size: 22px;">Event Details</h2>
                            
                            <table style="width: 100%; border-collapse: collapse;">
                                <tr>
                                    <td style="padding: 12px; border-bottom: 1px solid #ecf0f1; font-weight: bold; color: #7f8c8d;">Event Type:</td>
                                    <td style="padding: 12px; border-bottom: 1px solid #ecf0f1; color: #2c3e50;">{event_type.replace('_', ' ').title()}</td>
                                </tr>
                                <tr>
                                    <td style="padding: 12px; border-bottom: 1px solid #ecf0f1; font-weight: bold; color: #7f8c8d;">Detected Class:</td>
                                    <td style="padding: 12px; border-bottom: 1px solid #ecf0f1; color: #2c3e50;">{event_class.upper()}</td>
                                </tr>
                                <tr>
                                    <td style="padding: 12px; border-bottom: 1px solid #ecf0f1; font-weight: bold; color: #7f8c8d;">Timestamp:</td>
                                    <td style="padding: 12px; border-bottom: 1px solid #ecf0f1; color: #2c3e50;">{timestamp}</td>
                                </tr>
                                <tr>
                                    <td style="padding: 12px; border-bottom: 1px solid #ecf0f1; font-weight: bold; color: #7f8c8d;">Height:</td>
                                    <td style="padding: 12px; border-bottom: 1px solid #ecf0f1; color: #2c3e50;">{height if height != 'N/A' else 'Not measured'} {'m' if height != 'N/A' else ''}</td>
                                </tr>
                                <tr>
                                    <td style="padding: 12px; font-weight: bold; color: #7f8c8d;">In ROI:</td>
                                    <td style="padding: 12px; color: #2c3e50;">{"✅ Yes" if in_roi else "❌ No"}</td>
                                </tr>
                            </table>
                        </td>
                    </tr>
                    
                    <!-- Feedback Section -->
                    <tr>
                        <td style="padding: 0 30px 30px 30px;">
                            <div style="background-color: #f8f9fa; border-radius: 8px; padding: 25px; text-align: center;">
                                <h3 style="color: #2c3e50; margin: 0 0 15px 0; font-size: 18px;">Was this detection correct?</h3>
                                <p style="color: #7f8c8d; margin: 0 0 20px 0; font-size: 14px;">Your feedback helps improve detection accuracy</p>
                                
                                <table role="presentation" style="margin: 0 auto;">
                                    <tr>
                                        <td style="padding: 0 10px;">
                                            <a href="{correct_url}" style="display: inline-block; background-color: #27ae60; color: white; text-decoration: none; padding: 15px 35px; border-radius: 5px; font-weight: bold; font-size: 16px;">
                                                ✅ Correct
                                            </a>
                                        </td>
                                        <td style="padding: 0 10px;">
                                            <a href="{incorrect_url}" style="display: inline-block; background-color: #e74c3c; color: white; text-decoration: none; padding: 15px 35px; border-radius: 5px; font-weight: bold; font-size: 16px;">
                                                ❌ Wrong
                                            </a>
                                        </td>
                                    </tr>
                                </table>
                            </div>
                        </td>
                    </tr>
                    
                    <!-- View All Events Button -->
                    <tr>
                        <td style="padding: 0 30px 30px 30px; text-align: center;">
                            <a href="{view_url}" style="display: inline-block; background-color: #3498db; color: white; text-decoration: none; padding: 12px 30px; border-radius: 5px; font-weight: bold;">
                                📹 View All Events
                            </a>
                        </td>
                    </tr>
                    
                    <!-- Footer -->
                    <tr>
                        <td style="background-color: #34495e; color: #ecf0f1; padding: 20px; text-align: center; font-size: 12px;">
                            <p style="margin: 0;">Child Detection System</p>
                            <p style="margin: 5px 0 0 0;">Automated detection notification</p>
                        </td>
                    </tr>
                    
                </table>
            </td>
        </tr>
    </table>
</body>
</html>
        """
        
        return html
    
    def _send_email(self, msg):
        """Send email via SMTP"""
        try:
            print(f"[EMAIL] Connecting to {self.smtp_server}:{self.smtp_port}...")
            
            # Connect to SMTP server
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()  # Enable TLS encryption
            
            print(f"[EMAIL] Logging in as {self.sender_email}...")
            server.login(self.sender_email, self.sender_password)
            
            print(f"[EMAIL] Sending email to {self.recipient_email}...")
            server.send_message(msg)
            server.quit()
            
            print(f"[EMAIL] ✅ Email sent successfully!")
            return True
            
        except smtplib.SMTPAuthenticationError:
            print("[EMAIL] ❌ Authentication failed - check email/password")
            return False
        except smtplib.SMTPException as e:
            print(f"[EMAIL] ❌ SMTP error: {e}")
            return False
        except Exception as e:
            print(f"[EMAIL] ❌ Error sending email: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_email(self):
        """Send a test email to verify configuration"""
        if not self.enabled:
            return False, "Email notifications are disabled"
        
        if not all([self.sender_email, self.sender_password, self.recipient_email]):
            return False, "Missing email credentials"
        
        try:
            # Create test message
            msg = MIMEText("This is a test email from your Child Detection System. If you received this, email notifications are working correctly!")
            msg['Subject'] = "🧪 Test Email - Detection System"
            msg['From'] = self.sender_email
            msg['To'] = self.recipient_email
            
            # Send
            success = self._send_email(msg)
            
            if success:
                return True, "Test email sent successfully"
            else:
                return False, "Failed to send test email"
                
        except Exception as e:
            return False, f"Error: {str(e)}"