// Common Hardcoded Secrets Pattern - For Cache Warming
// This pattern should be detected as a security vulnerability

// VULNERABLE: Hardcoded API keys
const OPENAI_API_KEY = "sk-1234567890abcdef";
const AWS_SECRET_KEY = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY";
const DATABASE_PASSWORD = "super_secret_password_123";

// VULNERABLE: Hardcoded credentials in connection string
const connectionString = "mongodb://admin:password123@localhost:27017/mydb";

// VULNERABLE: Hardcoded JWT secret
const JWT_SECRET = "my-super-secret-jwt-key-12345";

// SAFE: Using environment variables
const apiKey = process.env.OPENAI_API_KEY;
const awsSecret = process.env.AWS_SECRET_ACCESS_KEY;
const dbPassword = process.env.DATABASE_PASSWORD;

// SAFE: Using configuration service
import { ConfigService } from "@nestjs/config";

class SecureService {
  constructor(private configService: ConfigService) {}

  getApiKey(): string {
    return this.configService.get<string>("OPENAI_API_KEY");
  }
}

// SAFE: Using secrets manager
import { SecretsManager } from "aws-sdk";

async function getSecret(secretName: string): Promise<string> {
  const client = new SecretsManager();
  const result = await client
    .getSecretValue({ SecretId: secretName })
    .promise();
  return result.SecretString || "";
}
