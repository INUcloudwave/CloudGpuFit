// TODO: Phase 3에서 AWS Pricing API 기반 동적 조회로 교체
export interface AwsRegionOption {
  code: string;
  label: string;
}

export const AWS_REGIONS: AwsRegionOption[] = [
  { code: "us-east-1", label: "버지니아 북부 (us-east-1)" },
  { code: "us-west-2", label: "오레곤 (us-west-2)" },
  { code: "eu-west-1", label: "아일랜드 (eu-west-1)" },
  { code: "eu-central-1", label: "프랑크푸르트 (eu-central-1)" },
  { code: "ap-northeast-1", label: "도쿄 (ap-northeast-1)" },
  { code: "ap-northeast-2", label: "서울 (ap-northeast-2)" },
  { code: "ap-southeast-1", label: "싱가포르 (ap-southeast-1)" },
  { code: "ap-south-1", label: "뭄바이 (ap-south-1)" },
  { code: "sa-east-1", label: "상파울루 (sa-east-1)" },
];

export const DEFAULT_AWS_REGION = "us-east-1";
