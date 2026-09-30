// Which regions the picker offers is still this static list — prices for
// whichever one is picked now come from a live AWS fetch (orchestrator/
// aws_pricing.py), but the region list itself isn't fetched from anywhere.
//
// All standard commercial regions (checked against the Price List Bulk API's
// region_index.json). Left out on purpose: GovCloud and China (separate
// partitions/accounts), and Local Zones / Wavelength zones (not full regions).
export interface AwsRegionOption {
  code: string;
  label: string;
}

export const AWS_REGIONS: AwsRegionOption[] = [
  // 북미
  { code: "us-east-1", label: "버지니아 북부 (us-east-1)" },
  { code: "us-east-2", label: "오하이오 (us-east-2)" },
  { code: "us-west-1", label: "캘리포니아 북부 (us-west-1)" },
  { code: "us-west-2", label: "오레곤 (us-west-2)" },
  { code: "ca-central-1", label: "캐나다 중부 (ca-central-1)" },
  { code: "ca-west-1", label: "캘거리 (ca-west-1)" },
  { code: "mx-central-1", label: "멕시코 중부 (mx-central-1)" },
  // 남미
  { code: "sa-east-1", label: "상파울루 (sa-east-1)" },
  // 유럽
  { code: "eu-west-1", label: "아일랜드 (eu-west-1)" },
  { code: "eu-west-2", label: "런던 (eu-west-2)" },
  { code: "eu-west-3", label: "파리 (eu-west-3)" },
  { code: "eu-central-1", label: "프랑크푸르트 (eu-central-1)" },
  { code: "eu-central-2", label: "취리히 (eu-central-2)" },
  { code: "eu-north-1", label: "스톡홀름 (eu-north-1)" },
  { code: "eu-south-1", label: "밀라노 (eu-south-1)" },
  { code: "eu-south-2", label: "스페인 (eu-south-2)" },
  // 아시아 태평양
  { code: "ap-northeast-1", label: "도쿄 (ap-northeast-1)" },
  { code: "ap-northeast-2", label: "서울 (ap-northeast-2)" },
  { code: "ap-northeast-3", label: "오사카 (ap-northeast-3)" },
  { code: "ap-east-1", label: "홍콩 (ap-east-1)" },
  { code: "ap-east-2", label: "타이베이 (ap-east-2)" },
  { code: "ap-southeast-1", label: "싱가포르 (ap-southeast-1)" },
  { code: "ap-southeast-2", label: "시드니 (ap-southeast-2)" },
  { code: "ap-southeast-3", label: "자카르타 (ap-southeast-3)" },
  { code: "ap-southeast-4", label: "멜버른 (ap-southeast-4)" },
  { code: "ap-southeast-5", label: "말레이시아 (ap-southeast-5)" },
  { code: "ap-southeast-6", label: "뉴질랜드 (ap-southeast-6)" },
  { code: "ap-southeast-7", label: "태국 (ap-southeast-7)" },
  { code: "ap-south-1", label: "뭄바이 (ap-south-1)" },
  { code: "ap-south-2", label: "하이데라바드 (ap-south-2)" },
  // 중동 · 아프리카
  { code: "me-south-1", label: "바레인 (me-south-1)" },
  { code: "me-central-1", label: "UAE (me-central-1)" },
  { code: "il-central-1", label: "텔아비브 (il-central-1)" },
  { code: "af-south-1", label: "케이프타운 (af-south-1)" },
];

export const DEFAULT_AWS_REGION = "us-east-1";
