import boto3
import sagemaker
import os
import time


def deploy_deepseek_on_aws():
    # Configurar sessão
    boto_session = boto3.session.Session()
    region = boto_session.region_name
    sess = sagemaker.Session()

    # Configuração da Implantação
    CONFIG = {
        "INSTANCE_TYPE": "ml.g6.12xlarge",  # Custo aproximado $5/hora - CUIDADO
        "ENV": {
            "HF_MODEL_ID": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            "TENSOR_PARALLEL_DEGREE": "max",  # Usa todas as GPUs disponíveis
            "OPTION_ENABLE_REASONING": "true",
            "OPTION_REASONING_PARSER": "deepseek_r1",
        },
        "IMAGE_NAME": "djl-inference:0.33.0-lmi15.0.0-cu128",
        # Substitua pelo ARN da sua role criada na AWS
        "BEDROCK_ROLE_ARN": "arn:aws:iam::YOUR_ACCOUNT_ID:role/YOUR_ROLE_NAME",
    }

    # Nomenclatura
    hf_model_id = CONFIG["ENV"]["HF_MODEL_ID"]
    model_name = hf_model_id.split("/")[-1]
    base_model_name = sagemaker.utils.name_from_base(model_name)
    current_region = os.environ.get("AWS_DEFAULT_REGION", region)

    # URI da imagem do container LMI
    inference_image_uri = f"763104351884.dkr.ecr.{current_region}.amazonaws.com/{CONFIG['IMAGE_NAME']}"

    # Criar modelo no SageMaker
    lmi_model = sagemaker.Model(
        image_uri=inference_image_uri,
        env=CONFIG["ENV"],
        role=CONFIG["BEDROCK_ROLE_ARN"],
        name=base_model_name,
    )

    endpoint_name = f"{base_model_name}-endpoint"

    print(f"Implantando modelo {model_name}...")
    print(f"Endpoint Name: {endpoint_name}")

    # Inicia o deploy (Pode levar 15 mins)
    lmi_model.deploy(
        initial_instance_count=1,
        instance_type=CONFIG["INSTANCE_TYPE"],
        container_startup_health_check_timeout=900,
        endpoint_name=endpoint_name,
    )

    print(f"Modelo implantado com sucesso no endpoint: {endpoint_name}")


if __name__ == "__main__":
    # deploy_deepseek_on_aws() # Descomente para executar (Cuidado com custos!)
    pass