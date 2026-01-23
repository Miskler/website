import aiohttp
import asyncio
import sys
from datetime import datetime
from collections import defaultdict

async def github_graphql_query(session, token, query):
    """
    Helper function to execute a GraphQL query asynchronously with error handling.
    """
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

async def main(token, username):
    async with aiohttp.ClientSession() as session:
        # 1. Получаем организации
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

        # 2. Получаем личные репозитории (где вы OWNER — всегда ADMIN)
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

        # 3. Получаем репозитории, где вы внешний коллаборатор
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

        # 4. Профиль с детальным contributionCalendar (для статистики по месяцам)
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

    # Обработка организаций
    org_nodes = org_data['data']['viewer']['organizations']['nodes']
    org_logins = [org['login'] for org in org_nodes]
    print("Ваши организации:")
    if org_nodes:
        for org in org_nodes:
            print(f"- Login: {org['login']}")
            print(f"  Name: {org.get('name') or 'N/A'}")
            print(f"  Description: {org.get('description') or 'None'}")
            print(f"  Avatar: {org.get('avatarUrl')}\n")
    else:
        print("  Организации не найдены.")

    print("\n" + "="*80 + "\n")

    # Получаем репозитории из организаций параллельно (с viewerPermission)
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

    # Собираем репозитории с правами ADMIN/MAINTAIN/WRITE
    filtered_repos = []

    # Личные (OWNER → ADMIN)
    for repo in personal_data['data']['viewer']['repositories']['edges']:
        filtered_repos.append(repo['node'])

    # Внешние коллаборации
    for repo in collab_data['data']['viewer']['repositories']['edges']:
        node = repo['node']
        if node['viewerPermission'] in ['ADMIN', 'MAINTAIN', 'WRITE']:
            filtered_repos.append(node)

    # Из организаций
    for result in org_repos_results:
        if isinstance(result, Exception):
            print(f"Ошибка при получении репозиториев организации: {result}")
            continue
        edges = result['data']['organization']['repositories']['edges']
        for repo in edges:
            node = repo['node']
            if node['viewerPermission'] in ['ADMIN', 'MAINTAIN', 'WRITE']:
                filtered_repos.append(node)

    # Вывод репозиториев
    print("Репозитории, где вы имеете права контрибьютора/мейнтейнера (ADMIN, MAINTAIN или WRITE):")
    if not filtered_repos:
        print("  Нет репозиториев с правами на запись (WRITE+).")
    else:
        for node in filtered_repos:
            languages = node['languages']
            total_size = languages['totalSize']
            lang_percent = {}
            if total_size > 0:
                for edge in languages['edges']:
                    lang_name = edge['node']['name']
                    lang_percent[lang_name] = round((edge['size'] / total_size) * 100, 2)

            fork_marker = "(FORK)" if node['isFork'] else ""
            permission_marker = f"({node.get('viewerPermission', 'ADMIN')})"
            print(f"- {node['owner']['login']}/{node['name']} {fork_marker} {permission_marker}")
            print(f"  Description: {node['description'] or 'None'}")
            print(f"  Stars: {node['stargazerCount']}, Open PRs: {node['pullRequests']['totalCount']}, Open Issues: {node['issues']['totalCount']}")
            print(f"  Languages: {lang_percent or 'None'}\n")

    print("="*80 + "\n")

    # Профиль и статистика контрибьюций по месяцам
    user = profile_data['data']['user']
    print("Информация о профиле:")
    print(f"- Login: {user['login']}")
    print(f"- Avatar URL: {user['avatarUrl']}")
    print(f"- Bio: {user['bio'] or 'None'}")
    print(f"- Created At: {user['createdAt']}")
    print(f"- Total Contributions (last year): {user['contributionsCollection']['contributionCalendar']['totalContributions']}")

    # Статистика по месяцам
    print("\nКонтрибьюции по месяцам (за последний год):")
    monthly_contributions = defaultdict(int)
    weeks = user['contributionsCollection']['contributionCalendar']['weeks']
    for week in weeks:
        for day in week['contributionDays']:
            date_str = day['date']  # YYYY-MM-DD
            count = day['contributionCount']
            month_key = datetime.strptime(date_str, "%Y-%m-%d").strftime("%B %Y")  # e.g., January 2026
            monthly_contributions[month_key] += count

    # Сортируем по дате (от старого к новому)
    sorted_months = sorted(monthly_contributions.items(), key=lambda x: datetime.strptime(x[0], "%B %Y"))
    if sorted_months:
        for month, count in sorted_months:
            print(f"- {month}: {count}")
    else:
        print("  Нет данных о контрибьюциях за последний год.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <github_token> <username>")
        sys.exit(1)
    token = sys.argv[1]
    username = sys.argv[2]
    asyncio.run(main(token, username))