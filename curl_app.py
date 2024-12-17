import subprocess
import os
import boto3

# S3 configuration
S3_BUCKET = "dtcml-outputs"  # Ganti dengan nama bucket S3
s3_client = boto3.client('s3')

def handler(event, context):
    try:
        # Step 1: Call API Gateway using curl
        api_url = "https://kpll0fgxze.execute-api.ap-southeast-3.amazonaws.com/test/predict/visualize_predictions?num_images=16"
        output_path = "/tmp/prediction.png"
        
        # Run curl command
        curl_command = ["curl", "-s", api_url, "--output", output_path]
        result = subprocess.run(curl_command, capture_output=True, text=True)
        
        # Check if curl command succeeded
        if result.returncode != 0:
            return {
                "statusCode": 500,
                "body": f"Error calling API Gateway: {result.stderr}"
            }
        
        # Step 2: Upload the file to S3
        s3_key = "prediction-lambda.png"
        s3_client.upload_file(output_path, S3_BUCKET, s3_key)
        
        # Step 3: Generate file URL
        file_url = f"https://{S3_BUCKET}.s3.amazonaws.com/{s3_key}"
        
        return {
            "statusCode": 200,
            "body": f"File successfully saved to S3: {file_url}"
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": f"Error: {str(e)}"
        }

if __name__ == "__main__":
    # Simulate AWS Lambda event and context
    event = {}
    context = {}
    
    response = handler(event, context)
    print("Response:")
    print(response)
