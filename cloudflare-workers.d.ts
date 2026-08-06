declare module "cloudflare:workers" {
  export const env: {
    DB?: any;
  };
}

interface Fetcher {
  fetch(input: RequestInfo | URL, init?: RequestInit): Promise<Response>;
}

interface D1Database {}
