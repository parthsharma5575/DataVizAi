#!/usr/bin/env python3
"""
Script to clear all data from the survey analysis platform
"""
import os
import shutil
from app import app, db
from models import UploadSession, ProcessingLog, ReportGeneration

def clear_all_data():
    """Clear all uploaded files, reports, and database records"""
    with app.app_context():
        try:
            # Clear database tables
            print("Clearing database records...")
            
            # Delete all report generation records
            ReportGeneration.query.delete()
            print("✓ Cleared report generation records")
            
            # Delete all processing logs
            ProcessingLog.query.delete() 
            print("✓ Cleared processing logs")
            
            # Update any stuck processing sessions to failed, then delete all
            UploadSession.query.filter(UploadSession.status.in_(['processing', 'pending'])).update(
                {'status': 'failed'}, synchronize_session=False
            )
            UploadSession.query.delete()
            print("✓ Cleared upload sessions")
            
            # Commit all database changes
            db.session.commit()
            print("✓ Database cleared successfully")
            
        except Exception as e:
            print(f"Error clearing database: {e}")
            db.session.rollback()
        
        # Clear uploaded files
        try:
            uploads_dir = 'uploads'
            if os.path.exists(uploads_dir):
                for filename in os.listdir(uploads_dir):
                    if filename != '.gitkeep':
                        file_path = os.path.join(uploads_dir, filename)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                            print(f"✓ Removed upload: {filename}")
                print("✓ Uploads directory cleared")
            
        except Exception as e:
            print(f"Error clearing uploads: {e}")
        
        # Clear generated reports
        try:
            reports_dir = 'reports'
            if os.path.exists(reports_dir):
                for filename in os.listdir(reports_dir):
                    if filename != '.gitkeep':
                        file_path = os.path.join(reports_dir, filename)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                            print(f"✓ Removed report: {filename}")
                print("✓ Reports directory cleared")
                
        except Exception as e:
            print(f"Error clearing reports: {e}")
        
        print("\n🎉 All data cleared successfully! Ready for fresh uploads.")

if __name__ == "__main__":
    clear_all_data()