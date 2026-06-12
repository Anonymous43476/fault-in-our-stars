import json

import pulumi
import pulumi_aws as aws

stack_name = pulumi.get_stack()

with open("pk.pub") as infile:
    public_key = infile.read()

with open("ssh_cidrs.json") as infile:
    cidrs = json.load(infile)

main = aws.ec2.DefaultVpc("default")

if stack_name == "dev":
    ssh_key = aws.ec2.KeyPair(
        "sec_agg_ssh", key_name="sec-agg-ssh", public_key=public_key
    )

    security_group = aws.ec2.SecurityGroup(
        "allow_ssh_home",
        name="allow_ssh_home",
        description="Allow SSH traffic from home computers",
        vpc_id=main.id,
        tags={
            "Name": "allow_ssh_home",
        },
    )

    results_bucket = aws.s3.BucketV2(
        "results_bucket",
        bucket="sam-decoder-results-2025",
        tags={"Name": "Decoder results"},
    )
else:
    ssh_key = aws.ec2.KeyPair("sec_agg_ssh", public_key=public_key)
    security_group = aws.ec2.SecurityGroup(
        "allow_ssh_home",
        description="Allow SSH traffic from home computers",
        vpc_id=main.id,
        tags={
            "Name": "allow_ssh_home",
        },
    )
    results_bucket = aws.s3.BucketV2(
        "YOUR NAME HERE",
        tags={"Name": "Decoder results"},
    )


for index, cidr in enumerate(cidrs):
    aws.vpc.SecurityGroupIngressRule(
        cidr.get("resource_name", f"allow_ssh_{index}"),
        security_group_id=security_group.id,
        cidr_ipv4=cidr["cidr"],
        ip_protocol="tcp",
        from_port=22,
        to_port=22,
        tags={
            "Name": cidr.get("name", f"SSH #{index}"),
            "Description": cidr.get("description", "Allows SSH"),
        },
    )

all_egress = aws.vpc.SecurityGroupEgressRule(
    "allow-all-egress",
    security_group_id=security_group.id,
    cidr_ipv4="0.0.0.0/0",
    ip_protocol="-1",
)


instance_role = aws.iam.Role(
    "decoder-instance-role",
    assume_role_policy=json.dumps(
        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Action": "sts:AssumeRole",
                    "Effect": "Allow",
                    "Sid": "",
                    "Principal": {
                        "Service": "ec2.amazonaws.com",
                    },
                }
            ],
        }
    ),
)

s3_policy_doc = results_bucket.arn.apply(
    lambda arn: aws.iam.get_policy_document(
        statements=[
            aws.iam.GetPolicyDocumentStatementArgs(
                actions=["s3:ListBucket", "s3:PutObject", "s3:GetObject"],
                resources=[arn, f"{arn}/*"],
            )
        ]
    ).json
)


if stack_name == "dev":
    s3_policy = aws.iam.Policy(
        "decoder_s3_access",
        name="decoder_s3_access",
        description="Allows the decoder instance to write to the results bucket",
        policy=s3_policy_doc,
    )
else:
    s3_policy = aws.iam.Policy(
        "decoder_s3_access",
        description="Allows the decoder instance to write to the results bucket",
        policy=s3_policy_doc,
    )

attachment = aws.iam.RolePolicyAttachment(
    "s3-attach", role=instance_role.name, policy_arn=s3_policy.arn
)

instance_profile = aws.iam.InstanceProfile("decoder-profile", role=instance_role.name)

instance = aws.ec2.Instance(
    "benchmark_server",
    ami="ami-04b4f1a9cf54c11d0",
    instance_type="c7i.8xlarge",
    key_name=ssh_key.key_name,
    vpc_security_group_ids=[security_group.id],
    iam_instance_profile=instance_profile.name,
    tags={
        "Name": (
            "benchmark_server"
            if stack_name == "dev"
            else f"benchmark_server_{stack_name}"
        ),
    },
)

ec2_termination_policy_doc = instance.arn.apply(
    lambda arn: aws.iam.get_policy_document(
        statements=[
            aws.iam.GetPolicyDocumentStatementArgs(
                actions=["ec2:StopInstances"],
                resources=[arn],
            )
        ]
    ).json
)

ec2_termination_policy = aws.iam.Policy(
    "decoder_self_termination",
    name=(
        "decoder_self_termination"
        if stack_name == "dev"
        else f"decoder_self_termination_{stack_name}"
    ),
    description="Allows the decoder to terminate itself",
    policy=ec2_termination_policy_doc,
)

termination_attachment = aws.iam.RolePolicyAttachment(
    "termination-attach",
    role=instance_role.name,
    policy_arn=ec2_termination_policy.arn,
)
