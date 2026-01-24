import aiohttp
import asyncio
from datetime import datetime
from collections import defaultdict
from async_lru import alru_cache


async def github_graphql_query(session, token, query):
    url = "https://api.github.com/graphql"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    payload = {"query": query}
    async with session.post(url, headers=headers, json=payload) as response:
        if response.status != 200:
            text = await response.text()
            raise Exception(f"HTTP {response.status}: {text}")
        data = await response.json()
        if 'errors' in data:
            error_msgs = "; ".join([err['message'] for err in data['errors']])
            raise Exception(f"GraphQL errors: {error_msgs}")
        return data

@alru_cache(ttl=240)
async def fetch_github_data(token: str, username: str) -> dict:
    """
    Единая асинхронная функция, которая собирает все требуемые данные о GitHub-профиле
    и возвращает их в виде словаря (готового к сериализации в JSON).
    """
    async with aiohttp.ClientSession() as session:
        # 1. Организации
        orgs_task = asyncio.create_task(github_graphql_query(session, token, """
        query {
          viewer {
            organizations(first: 100) {
              nodes {
                login
                name
                description
                avatarUrl
              }
            }
          }
        }
        """))

        # 2. Личные репозитории (OWNER)
        personal_task = asyncio.create_task(github_graphql_query(session, token, """
        query {
          viewer {
            repositories(first: 100, affiliations: [OWNER]) {
              edges {
                node {
                  name
                  owner { login }
                  description
                  stargazerCount
                  pullRequests(states: OPEN) { totalCount }
                  issues(states: OPEN) { totalCount }
                  isFork
                  viewerPermission
                  languages(first: 10, orderBy: {field: SIZE, direction: DESC}) {
                    edges { size node { name } }
                    totalSize
                  }
                }
              }
            }
          }
        }
        """))

        # 3. Репозитории с внешней коллаборацией
        collab_task = asyncio.create_task(github_graphql_query(session, token, """
        query {
          viewer {
            repositories(first: 100, affiliations: [COLLABORATOR]) {
              edges {
                node {
                  name
                  owner { login }
                  description
                  stargazerCount
                  pullRequests(states: OPEN) { totalCount }
                  issues(states: OPEN) { totalCount }
                  isFork
                  viewerPermission
                  languages(first: 10, orderBy: {field: SIZE, direction: DESC}) {
                    edges { size node { name } }
                    totalSize
                  }
                }
              }
            }
          }
        }
        """))

        # 4. Профиль + полный contributionCalendar
        profile_task = asyncio.create_task(github_graphql_query(session, token, f"""
        query {{
          user(login: "{username}") {{
            login
            avatarUrl
            bio
            createdAt
            contributionsCollection {{
              contributionCalendar {{
                totalContributions
                weeks {{
                  contributionDays {{
                    contributionCount
                    date
                  }}
                }}
              }}
            }}
          }}
        }}
        """))

        org_data, personal_data, collab_data, profile_data = await asyncio.gather(
            orgs_task, personal_task, collab_task, profile_task
        )

    # Организации
    org_nodes = org_data['data']['viewer']['organizations']['nodes']
    organizations = []
    org_logins = [org['login'] for org in org_nodes]
    for org in org_nodes:
        organizations.append({
            "login": org['login'],
            "name": org.get('name') or None,
            "description": org.get('description') or None,
            "avatar_url": org.get('avatarUrl') or None
        })

    # Репозитории из организаций (параллельно)
    async with aiohttp.ClientSession() as session:
        org_repo_tasks = []
        for org_login in org_logins:
            query = f"""
            query {{
              organization(login: "{org_login}") {{
                repositories(first: 100) {{
                  edges {{
                    node {{
                      name
                      owner {{ login }}
                      description
                      stargazerCount
                      pullRequests(states: OPEN) {{ totalCount }}
                      issues(states: OPEN) {{ totalCount }}
                      isFork
                      viewerPermission
                      languages(first: 10, orderBy: {{field: SIZE, direction: DESC}}) {{
                        edges {{ size node {{ name }} }}
                        totalSize
                      }}
                    }}
                  }}
                }}
              }}
            }}
            """
            org_repo_tasks.append(github_graphql_query(session, token, query))

        org_repos_results = await asyncio.gather(*org_repo_tasks, return_exceptions=True)

    # Сбор отфильтрованных репозиториев
    repositories = []

    # Личные
    for repo in personal_data['data']['viewer']['repositories']['edges']:
        node = repo['node']
        repositories.append(_process_repo_node(node, default_permission="ADMIN"))

    # Внешние коллаборации
    for repo in collab_data['data']['viewer']['repositories']['edges']:
        node = repo['node']
        if node['viewerPermission'] in ['ADMIN', 'MAINTAIN', 'WRITE']:
            repositories.append(_process_repo_node(node))

    # Из организаций
    for result in org_repos_results:
        if isinstance(result, Exception):
            continue
        edges = result['data']['organization']['repositories']['edges']
        for repo in edges:
            node = repo['node']
            if node['viewerPermission'] in ['ADMIN', 'MAINTAIN', 'WRITE']:
                repositories.append(_process_repo_node(node))

    # Профиль
    user = profile_data['data']['user']
    profile = {
        "login": user['login'],
        "avatar_url": user['avatarUrl'],
        "bio": user['bio'] or None,
        "created_at": user['createdAt'],
        "total_contributions_last_year": user['contributionsCollection']['contributionCalendar']['totalContributions']
    }

    # Контрибьюции по месяцам
    monthly_contributions = defaultdict(int)
    weeks = user['contributionsCollection']['contributionCalendar']['weeks']
    for week in weeks:
        for day in week['contributionDays']:
            date_str = day['date']
            count = day['contributionCount']
            month_key = datetime.strptime(date_str, "%Y-%m-%d").strftime("%B %Y")
            monthly_contributions[month_key] += count

    # Сортируем по дате
    sorted_monthly = dict(sorted(monthly_contributions.items(), key=lambda x: datetime.strptime(x[0], "%B %Y")))

    # Итоговый результат
    return {
        "organizations": organizations,
        "repositories": repositories,
        "profile": profile,
        "monthly_contributions": sorted_monthly
    }

def _process_repo_node(node: dict, default_permission: str = None) -> dict:
    """Вспомогательная функция для обработки узла репозитория."""
    languages = node['languages']
    total_size = languages['totalSize']
    lang_percent = {}
    if total_size > 0:
        for edge in languages['edges']:
            name = edge['node']['name']
            lang_percent[name] = round((edge['size'] / total_size) * 100, 2)

    permission = node.get('viewerPermission') or default_permission

    return {
        "full_name": f"{node['owner']['login']}/{node['name']}",
        "owner": node['owner']['login'],
        "name": node['name'],
        "description": node['description'] or None,
        "stars": node['stargazerCount'],
        "open_pull_requests": node['pullRequests']['totalCount'],
        "open_issues": node['issues']['totalCount'],
        "is_fork": node['isFork'],
        "permission": permission,
        "languages_percent": lang_percent
    }
