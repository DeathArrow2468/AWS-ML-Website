import boto3

sagemaker = boto3.client("sagemaker")

PIPELINE_NAME = "ML-Auto-Preprocess-Pipeline"

def lambda_handler(event, context):
    record = event["Records"][0]
    bucket = record["s3"]["bucket"]["name"]
    key = record["s3"]["object"]["key"]

    print(f"New upload detected: s3://{bucket}/{key}")

    sagemaker.start_pipeline_execution(
        PipelineName=PIPELINE_NAME,
        PipelineParameters=[
            {
                "Name": "InputDataS3Uri",
                "Value": f"s3://{bucket}/user-uploads/"
            }
        ]
    )

    return {
        "statusCode": 200,
        "body": "Pipeline execution started"
    }
