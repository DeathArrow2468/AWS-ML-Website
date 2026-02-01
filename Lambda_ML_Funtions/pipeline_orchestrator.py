import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.parameters import ParameterString
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput

# ----------------------------------
# Session & role
# ----------------------------------
sess = sagemaker.session.Session()
role = sagemaker.get_execution_role()

# ----------------------------------
# Pipeline parameter
# ----------------------------------
input_s3 = ParameterString(
    name="InputDataS3Uri",
    default_value="s3://ml-userdata-website/user-uploads/"
)

# ----------------------------------
# Shared processor
# ----------------------------------
processor = SKLearnProcessor(
    framework_version="1.2-1",
    role=role,
    instance_type="ml.m5.large",
    instance_count=1,
)

# ----------------------------------
# STEP 1: PreprocessData
# ----------------------------------
step_process = ProcessingStep(
    name="PreprocessData",
    processor=processor,
    inputs=[
        ProcessingInput(
            source=input_s3,
            destination="/opt/ml/processing/input"
        )
    ],
    outputs=[
        ProcessingOutput(
            source="/opt/ml/processing/train",
            destination="s3://ml-userdata-website/processed/train/"
        ),
        ProcessingOutput(
            source="/opt/ml/processing/validation",
            destination="s3://ml-userdata-website/processed/validation/"
        ),
    ],
    code="preprocess.py",
)

# ----------------------------------
# STEP 2: RunModelComparison
# ----------------------------------
model_eval_step = ProcessingStep(
    name="RunModelComparison",
    processor=processor,
    inputs=[
        ProcessingInput(
            source=step_process.properties
                .ProcessingOutputConfig
                .Outputs[0]
                .S3Output
                .S3Uri,
            destination="/opt/ml/processing/train"
        ),
        ProcessingInput(
            source=step_process.properties
                .ProcessingOutputConfig
                .Outputs[1]
                .S3Output
                .S3Uri,
            destination="/opt/ml/processing/validation"
        ),
        ProcessingInput(
            source="s3://ml-aws-manavhm/code/",
            destination="/opt/ml/processing/code"
        ),
    ],
    outputs=[
        ProcessingOutput(
            source="/opt/ml/processing/results",
            destination="s3://ml-code-website/results/"
        )
    ],
    code="run_model_comparison.py",
)


# ----------------------------------
# PIPELINE
# ----------------------------------
pipeline = Pipeline(
    name="ML-Auto-Preprocess-Pipeline",
    parameters=[input_s3],
    steps=[step_process, model_eval_step],
    sagemaker_session=sess,
)

pipeline.upsert(role_arn=role)
