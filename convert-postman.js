import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const postmanFile = process.argv[2];
if (!postmanFile) {
  console.error('Usage: node convert-postman.js <postman-collection.json>');
  process.exit(1);
}

const postmanPath = path.resolve(postmanFile);
const postman = JSON.parse(fs.readFileSync(postmanPath, 'utf8'));

const openapi = {
  openapi: '3.0.0',
  info: {
    title: postman.info.name || 'API',
    version: '1.0.0',
    description: postman.info.description || ''
  },
  servers: [],
  paths: {},
  components: {
    securitySchemes: {
      bearerAuth: {
        type: 'http',
        scheme: 'bearer'
      }
    }
  }
};

function processItem(item, basePath = '') {
  if (item.item) {
    item.item.forEach(subItem => processItem(subItem, basePath));
  } else if (item.request) {
    const req = item.request;
    const method = req.method.toLowerCase();
    
    let pathStr = '';
    if (req.url) {
      if (typeof req.url === 'string') {
        pathStr = req.url;
      } else if (req.url.path) {
        pathStr = '/' + req.url.path.join('/');
      }
    }
    
    pathStr = pathStr.replace(/\{\{[^}]+\}\}/g, (match) => {
      const varName = match.slice(2, -2);
      return `{${varName}}`;
    });
    
    if (!pathStr.startsWith('/')) {
      pathStr = '/' + pathStr;
    }
    
    if (!openapi.paths[pathStr]) {
      openapi.paths[pathStr] = {};
    }
    
    const operation = {
      summary: item.name || '',
      description: req.description || '',
      responses: {
        '200': {
          description: 'Successful response',
          content: {
            'application/json': {
              schema: {
                type: 'object'
              }
            }
          }
        }
      }
    };
    
    if (req.auth && req.auth.type === 'bearer') {
      operation.security = [{ bearerAuth: [] }];
    }
    
    if (req.header && req.header.length > 0) {
      operation.parameters = req.header
        .filter(h => !h.disabled)
        .map(h => ({
          name: h.key,
          in: 'header',
          schema: { type: 'string' },
          description: h.description || ''
        }));
    }
    
    if (req.url && req.url.query) {
      if (!operation.parameters) operation.parameters = [];
      req.url.query.forEach(q => {
        operation.parameters.push({
          name: q.key,
          in: 'query',
          schema: { type: 'string' },
          description: q.description || ''
        });
      });
    }
    
    if (req.body && req.body.mode === 'raw' && ['post', 'put', 'patch'].includes(method)) {
      operation.requestBody = {
        content: {
          'application/json': {
            schema: {
              type: 'object',
              example: req.body.raw
            }
          }
        }
      };
    }
    
    openapi.paths[pathStr][method] = operation;
  }
}

if (postman.item) {
  postman.item.forEach(item => processItem(item));
}

const outputPath = postmanPath.replace(/\.json$/, '-openapi.json');
fs.writeFileSync(outputPath, JSON.stringify(openapi, null, 2));
console.log(`OpenAPI spec generated: ${outputPath}`);
