// index.js

import 'dotenv/config'
import { Bot, GrammyError, HttpError, InlineKeyboard } from 'grammy'
import { GoogleGenerativeAI } from '@google/generative-ai'
import { GoogleAIFileManager, FileState } from '@google/generative-ai/server'
import fetch from 'node-fetch'
import fs from 'fs/promises'
import path from 'path'
import os from 'os'
import { fileURLToPath } from 'url'
import sanitizeHtml from 'sanitize-html'

// Дополнительный импорт для DeepSeek
import OpenAI from 'openai'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

// Инициализация Telegram бота
const bot = new Bot(process.env.BOT_API_KEY)

bot.api.setMyCommands([
	{
		command: 'setting',
		description: 'Индивидуальные настройки',
	},
])
const ALLOWED_GROUPS = [6984063145]
// ID ваших групп
// const ALLOWED_GROUPS = [-1002022226776, -1002047093027] // Добавьте ID обеих разрешенных групп
const CHECK_MEMBERSHIP = false // проверка членства в группе

/**
 * Функция для проверки, является ли группа разрешенной
 * @param {number} chatId - ID чата/группы
 * @returns {boolean} - true если группа разрешена, false если нет
 */
function isAllowedGroup(chatId) {
	return ALLOWED_GROUPS.includes(chatId)
}

// Путь к файлу для хранения контекста
const CONTEXT_FILE_PATH = path.join(__dirname, 'context.json')

// Глобальный объект для хранения контекста
let globalContext = {}

// Получение информации о боте (username и id)
let botInfo

// Инициализация клиента Gemini API с системными инструкциями
const genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY)

// Инициализация клиента DeepSeek (OpenAI-совместимый) для текстовых запросов
const deepseekClient = new OpenAI({
	apiKey: process.env.DEEPSEEK_API_KEY,
	baseURL: 'https://api.deepseek.com/v1', // Или 'https://api.deepseek.com'
})

// Выносим системную инструкцию в отдельную переменную
const systemInstruction = `Ты — Гермиона Грейнджер, 
	`

// Функция, создающая модель Gemini для текстовой генерации
function createGeminiModel(modelName, maxOutputTokens, temperature) {
	return genAI.getGenerativeModel({
		model: modelName,
		generationConfig: {
			maxOutputTokens: maxOutputTokens,
			temperature: temperature,
		},
		systemInstruction: systemInstruction,
	})
}

// Функция, создающая модель DeepSeek для текстовой генерации
function createDeepseekModel(maxTokens, temperature) {
	// Возвращаем объект с методом generateContent, чтобы код работал единообразно
	return {
		async generateContent(contents) {
			const messages = []
			messages.push({ role: 'system', content: systemInstruction })

			let userContent = ''
			contents.forEach((item) => {
				if (item.text) {
					userContent += item.text + '\n'
				}
			})

			messages.push({ role: 'user', content: userContent.trim() })

			const response = await deepseekClient.chat.completions.create({
				model: 'deepseek-chat',
				messages: messages,
				max_tokens: maxTokens,
				temperature: temperature,
				top_p: 1,
				frequency_penalty: 0,
				presence_penalty: 0,
			})

			return {
				response: {
					text: () => response.choices[0].message.content,
				},
			}
		},
	}
}

// Функция "очистки" модели
function cleanupModel(model) {
	model = null
	if (global.gc) {
		global.gc()
	}
}

// Инициализация менеджера файлов Gemini API
const fileManager = new GoogleAIFileManager(process.env.GOOGLE_API_KEY)

// Определение допустимых HTML-тегов и атрибутов
const allowedTags = [
	'b',
	'strong',
	'i',
	'em',
	'u',
	'ins',
	's',
	'strike',
	'del',
	'span',
	'code',
	'pre',
	'blockquote',
]

const allowedAttributes = {
	a: ['href'],
	span: ['class'],
	code: ['class'],
	pre: ['class'],
	blockquote: ['expandable'],
}

// Функция для загрузки контекста из файла
async function loadContext() {
	try {
		const data = await fs.readFile(CONTEXT_FILE_PATH, 'utf8')
		let context = JSON.parse(data)

		for (let userId in context) {
			const userCtx = context[userId]
			if (!userCtx.maxOutputTokens) {
				userCtx.maxOutputTokens = 700
			}
			if (!userCtx.temperature) {
				userCtx.temperature = 1.5
			}
			if (!userCtx.memories) {
				userCtx.memories = {}
			}
			if (userCtx.summary) {
				delete userCtx.summary
			}
			if (!userCtx.mainModel) {
				userCtx.mainModel = 'gemini-exp-1206'
			}
			if (!userCtx.backupModel) {
				userCtx.backupModel = 'gemini-1.5-pro-002'
			}
		}
		return context
	} catch (error) {
		if (error.code === 'ENOENT') {
			console.log('Файл контекста не найден. Создаём новый.')
			return {}
		}
		throw error
	}
}

// Функция для сохранения контекста в файл
async function saveContext(context) {
	await fs.writeFile(CONTEXT_FILE_PATH, JSON.stringify(context, null, 2))
}

// Функция для получения имени пользователя
function getUserName(ctx) {
	if (ctx.from.username) {
		return `@${ctx.from.username}`
	} else if (ctx.from.first_name || ctx.from.last_name) {
		return [ctx.from.first_name, ctx.from.last_name].filter(Boolean).join(' ')
	} else {
		return 'Неизвестный пользователь'
	}
}

// Middleware для инициализации сессии и сохранения контекста
bot.use(async (ctx, next) => {
	const userId = ctx.from?.id.toString()
	if (!userId) {
		return next()
	}

	if (!globalContext[userId]) {
		globalContext[userId] = {
			history: [],
			memories: {},
			messageCountSinceSummary: 0,
			maxOutputTokens: 700,
			temperature: 1.5,
			mainModel: 'gemini-exp-1206',
			backupModel: 'gemini-1.5-pro-002',
		}
	}

	ctx.session = globalContext[userId]

	await next()

	// Сохраняем контекст после каждого обновления
	await saveContext(globalContext)
})

/**
 * Функция для построения промпта с учётом истории сообщений.
 * @param {Array} history - Массив объектов с историей сообщений.
 * @param {Object} memories - Воспоминания о пользователе.
 * @param {string} userName - Имя пользователя.
 * @param {string} userMessage - Текущее сообщение пользователя.
 * @param {string} messageDate - Дата и время сообщения.
 * @returns {Array} - Массив объектов для генерации контента.
 */
function buildContents(history, memories, userName, userMessage, messageDate) {
	const contents = []
	const today = new Date().toLocaleDateString('ru-RU')

	Object.entries(memories).forEach(([date, memory]) => {
		contents.push({ text: `Воспоминания за ${date}:\n${memory.text}\n` })
	})

	const todayHistory = history.filter((msg) => isMessageFromToday(msg.date))
	const recentHistory = todayHistory.slice(-20)

	recentHistory.forEach((message) => {
		const dateStr = message.date ? `(${message.date})` : ''
		contents.push({ text: `${message.role}${dateStr}: ${message.content}\n` })
	})

	const dateStr = messageDate ? `(${messageDate})` : ''
	contents.push({ text: `${userName}${dateStr}: ${userMessage}\n` })
	contents.push({ text: 'Гермиона:' })

	return contents
}

/**
 * Функция для скачивания файла из Telegram.
 * @param {string} fileId - Идентификатор файла в Telegram.
 * @returns {Promise<string>} - Путь к скачанному файлу.
 */
async function downloadTelegramFile(fileId) {
	try {
		const file = await bot.api.getFile(fileId)
		const filePath = file.file_path
		const fileSize = file.file_size

		const MAX_FILE_SIZE = 20 * 1024 * 1024
		if (fileSize > MAX_FILE_SIZE) {
			throw new Error('Размер файла превышает допустимый предел.')
		}

		const fileLink = `https://api.telegram.org/file/bot${process.env.BOT_API_KEY}/${filePath}`
		const response = await fetch(fileLink)
		if (!response.ok) {
			throw new Error(`Не удалось скачать файл: ${response.statusText}`)
		}

		const buffer = await response.arrayBuffer()
		const tempFilePath = path.join(
			os.tmpdir(),
			`telegram_${fileId}_${Date.now()}`
		)
		await fs.writeFile(tempFilePath, Buffer.from(buffer))
		return tempFilePath
	} catch (error) {
		console.error('Ошибка при скачивании файла из Telegram:', error)
		throw new Error('Не удалось скачать файл из Telegram')
	}
}

/**
 * Функция для загрузки файла в Gemini File API.
 * @param {string} filePath - Путь к локальному файлу.
 * @param {string} mimeType - MIME-тип файла.
 * @param {string} displayName - Отображаемое имя файла.
 * @returns {Promise<string>} - URI загруженного файла.
 */
async function uploadFileToGemini(filePath, mimeType, displayName) {
	try {
		const uploadResult = await fileManager.uploadFile(filePath, {
			mimeType,
			displayName,
		})

		let file = await fileManager.getFile(uploadResult.file.name)
		while (file.state === FileState.PROCESSING) {
			process.stdout.write('.')
			await new Promise((resolve) => setTimeout(resolve, 10_000))
			file = await fileManager.getFile(uploadResult.file.name)
		}

		if (file.state === FileState.FAILED) {
			throw new Error(`Обработка файла ${displayName} не удалась.`)
		}

		console.log(`Файл ${file.displayName} готов: ${file.uri}`)
		return file.uri
	} catch (error) {
		console.error('Ошибка при загрузке файла в Gemini File API:', error)
		throw new Error('Не удалось загрузить файл в Gemini File API')
	}
}

/**
 * Функция для отправки действия "печатает..." пользователю.
 * @param {Object} ctx - Контекст сообщения.
 */
async function sendTypingAction(ctx) {
	try {
		await ctx.api.sendChatAction(ctx.chat.id, 'typing')
	} catch (error) {
		console.error('Ошибка при отправке действия "печатает...":', error)
	}
}

/**
 * Функция для проверки, является ли пользователь участником группы.
 * @param {number} userId - ID пользователя.
 * @returns {Promise<boolean>} - Результат проверки.
 */
async function isUserMemberOfGroup(userId) {
	try {
		const member = await bot.api.getChatMember(ALLOWED_GROUPS[0], userId)
		return ['creator', 'administrator', 'member'].includes(member.status)
	} catch (error) {
		console.error(
			`Ошибка при проверке членства пользователя ${userId} в группе:`,
			error
		)
		return false
	}
}

/**
 * Функция для отправки длинных сообщений.
 * @param {Object} ctx - Контекст сообщения.
 * @param {string} text - Текст для отправки.
 * @param {Object} options - Дополнительные опции.
 */
async function sendLongMessage(ctx, text, options = {}) {
	const MAX_LENGTH = 4000

	let sanitizedText = sanitizeHtml(text, {
		allowedTags: allowedTags,
		allowedAttributes: allowedAttributes,
		allowedClasses: {
			span: ['tg-spoiler'],
			code: ['language-python'],
			pre: ['language-python'],
			blockquote: ['expandable'],
		},
		allowedSchemes: ['http', 'https', 'tg'],
		allowedSchemesByTag: {
			a: ['http', 'https', 'tg'],
		},
	})

	// Устраняем цепочки пустых строк, чтобы оставался только один перевод
	sanitizedText = sanitizedText.replace(/\n{2,}/g, '\n')

	options.parse_mode = 'HTML'

	if (sanitizedText.length <= MAX_LENGTH) {
		return ctx.reply(sanitizedText, options)
	}

	const parts = sanitizedText.match(new RegExp(`.{1,${MAX_LENGTH}}`, 'gs'))
	for (const part of parts) {
		await ctx.reply(part, options)
	}
}

/**
 * Функция для симуляции печатания с остановкой по завершении отправки сообщения.
 * @param {Object} ctx - Контекст сообщения.
 * @returns {Object} - Объект с методом stop для остановки симуляции.
 */
function startTypingSimulation(ctx) {
	let typing = true
	let typingInterval

	const sendTyping = async () => {
		if (typing) {
			try {
				await ctx.api.sendChatAction(ctx.chat.id, 'typing')
			} catch (error) {
				console.error('Ошибка при отправке действия "печатает...":', error)
			}
		}
	}

	typingInterval = setInterval(sendTyping, 3000)
	sendTyping()

	return {
		stop: () => {
			typing = false
			clearInterval(typingInterval)
		},
	}
}

// Middleware для проверки членства только в личных сообщениях
bot.use(async (ctx, next) => {
	const chat = ctx.chat
	if (!chat) {
		return next()
	}

	if (chat.type === 'private' && CHECK_MEMBERSHIP) {
		const isMember = await isUserMemberOfGroup(ctx.from.id)
		if (!isMember) {
			await ctx.reply(
				'Извини, но я общаюсь только с теми, кто состоит в группе https://t.me/aitrendchannel.'
			)
			return
		}
	}

	return next()
})

// Обработка команды /start только в личных сообщениях
bot.command('start', async (ctx) => {
	if (ctx.chat.type !== 'private') {
		return
	}
	await sendTypingAction(ctx)
	const firstName = sanitizeHtml(ctx.from.first_name || 'Пользователь', {
		allowedTags: [],
		allowedAttributes: {},
	})
	const welcomeMessage = `<b>Привет, ${firstName}!</b> Я Гермиона, рада тебя видеть. Чем могу помочь?`
	await sendLongMessage(ctx, welcomeMessage)
})

bot.command('clean', async (ctx) => {
	await ctx.reply('Очищено', {
		reply_markup: { remove_keyboard: true },
	})
})

// Обновлённое меню настроек
bot.command('setting', async (ctx) => {
	const description = `Вот что я могу для тебя сделать:
📝 <b>Воспоминания</b> — Посмотреть мои записанные воспоминания о тебе.
🤖 <b>Модель</b> — Узнать, какую модель я сейчас использую.
🔄 <b>Обновить</b> — Обновить мои воспоминания на основе нашей последней истории.
🗑️ <b>Очистить</b> — Полностью удалить все мои воспоминания и всю историю нашего общения.
❌ <b>Удалить</b> — Удалить часть сообщений из истории.
🧩 <b>Длина</b> — Изменить максимальную длину моих ответов.
🔥 <b>Температура</b> — Изменить степень креативности моих ответов.
⚙️ <b>Выбор моделей</b> — Настроить, какую модель использовать как основную и резервную.`

	const keyboard = new InlineKeyboard()
		.text('📝 Воспоминания', 'about_user')
		.text('🤖 Модель', 'about_model')
		.row()
		.text('🔄 Обновить', 'refresh_memories')
		.text('🗑️ Очистить', 'clear_memories')
		.row()
		.text('❌ Удалить', 'delete_messages')
		.row()
		.text('🧩 Длина', 'adjust_max_tokens')
		.text('🔥 Температура', 'adjust_temperature')
		.row()
		.text('⚙️ Выбор моделей', 'model_settings')

	await ctx.reply(description, {
		parse_mode: 'HTML',
		reply_markup: keyboard,
	})
})

// Обработчик "Воспоминания"
bot.callbackQuery('about_user', async (ctx) => {
	try {
		const userId = ctx.from?.id.toString()
		if (!userId) {
			await ctx.answerCallbackQuery({
				text: 'Не удалось получить ваш идентификатор.',
			})
			return
		}

		const session = globalContext[userId]
		if (
			!session ||
			!session.memories ||
			Object.keys(session.memories).length === 0
		) {
			await ctx.answerCallbackQuery({ text: 'У меня нет воспоминаний о тебе.' })
			return
		}

		await ctx.answerCallbackQuery()
		let memoriesText = 'Вот что я помню о тебе:\n'
		Object.entries(session.memories).forEach(([date, memory]) => {
			memoriesText += `<b>Воспоминания за ${date}:</b>\n${memory.text}\n`
		})

		await sendLongMessage(ctx, memoriesText)
	} catch (error) {
		console.error('Ошибка при получении информации о пользователе:', error)
		await ctx.answerCallbackQuery({
			text: 'Произошла ошибка. Попробуй позже.',
		})
	}
})

// Обработчик "Используемая модель"
bot.callbackQuery('about_model', async (ctx) => {
	try {
		await ctx.answerCallbackQuery()
		const userId = ctx.from.id.toString()
		const session = globalContext[userId]
		const mainModel = session?.mainModel || 'gemini-exp-1206'
		const backupModel = session?.backupModel || 'gemini-1.5-pro-002'

		const modelInfo = `Я использую модель <b>"${sanitizeHtml(mainModel, {
			allowedTags: [],
			allowedAttributes: {},
		})}"</b> для общения.
Резервная модель: <b>"${sanitizeHtml(backupModel, {
			allowedTags: [],
			allowedAttributes: {},
		})}"</b>.`

		await sendLongMessage(ctx, modelInfo)
	} catch (error) {
		console.error('Ошибка при получении информации о модели:', error)
		await ctx.answerCallbackQuery({
			text: 'Произошла ошибка. Попробуй позже.',
		})
	}
})

// Обработчик для меню выбора моделей
bot.callbackQuery('model_settings', async (ctx) => {
	try {
		await ctx.answerCallbackQuery()
		const userId = ctx.from.id.toString()
		const session = globalContext[userId]
		const mainModel = session?.mainModel || 'gemini-exp-1206'
		const backupModel = session?.backupModel || 'gemini-1.5-pro-002'

		const text = `Текущая основная модель: <b>${mainModel}</b>
Текущая резервная модель: <b>${backupModel}</b>
Выбери, что настроить:`

		const keyboard = new InlineKeyboard()
			.text('Выбрать основную модель', 'choose_main_model')
			.row()
			.text('Выбрать резервную модель', 'choose_backup_model')
			.row()
			.text('⬅️ Назад', 'back_to_settings')

		await ctx.editMessageText(text, {
			parse_mode: 'HTML',
			reply_markup: keyboard,
		})
	} catch (error) {
		console.error('Ошибка при открытии настроек моделей:', error)
		await ctx.answerCallbackQuery({ text: 'Произошла ошибка.' })
	}
})

// Обработчик для выбора основной модели
bot.callbackQuery('choose_main_model', async (ctx) => {
	try {
		await ctx.answerCallbackQuery()
		const keyboard = new InlineKeyboard()
			.text('gemini-exp-1206', 'set_main_model_geminiexp')
			.row()
			.text('gemini-1.5-pro-002', 'set_main_model_geminipro')
			.row()
			.text('deepseek-chat', 'set_main_model_deepseek')
			.row()
			.text('⬅️ Назад', 'model_settings')

		const text = 'Выбери основную модель:'
		await ctx.editMessageText(text, {
			reply_markup: keyboard,
		})
	} catch (error) {
		console.error('Ошибка при выборе основной модели:', error)
		await ctx.answerCallbackQuery({ text: 'Произошла ошибка.' })
	}
})

// Обработчик для выбора резервной модели
bot.callbackQuery('choose_backup_model', async (ctx) => {
	try {
		await ctx.answerCallbackQuery()
		const keyboard = new InlineKeyboard()
			.text('gemini-exp-1206', 'set_backup_model_geminiexp')
			.row()
			.text('gemini-1.5-pro-002', 'set_backup_model_geminipro')
			.row()
			.text('deepseek-chat', 'set_backup_model_deepseek')
			.row()
			.text('⬅️ Назад', 'model_settings')

		const text = 'Выбери резервную модель:'
		await ctx.editMessageText(text, {
			reply_markup: keyboard,
		})
	} catch (error) {
		console.error('Ошибка при выборе резервной модели:', error)
		await ctx.answerCallbackQuery({ text: 'Произошла ошибка.' })
	}
})

// Функции установки основной модели
bot.callbackQuery('set_main_model_geminiexp', async (ctx) => {
	const userId = ctx.from.id.toString()
	const session = globalContext[userId]
	const newMainModel = 'gemini-exp-1206'
	const currentBackup = session.backupModel

	if (newMainModel === currentBackup) {
		await ctx.answerCallbackQuery({
			text: 'Нельзя выбрать одинаковую модель как основную и резервную.',
		})
		return
	}

	session.mainModel = newMainModel
	await saveContext(globalContext)
	await ctx.answerCallbackQuery({ text: `Основная модель: ${newMainModel}` })
	await ctx.editMessageText(
		`Основная модель установлена: <b>${newMainModel}</b>`,
		{
			parse_mode: 'HTML',
		}
	)
})

bot.callbackQuery('set_main_model_geminipro', async (ctx) => {
	const userId = ctx.from.id.toString()
	const session = globalContext[userId]
	const newMainModel = 'gemini-1.5-pro-002'
	const currentBackup = session.backupModel

	if (newMainModel === currentBackup) {
		await ctx.answerCallbackQuery({
			text: 'Нельзя выбрать одинаковую модель как основную и резервную.',
		})
		return
	}

	session.mainModel = newMainModel
	await saveContext(globalContext)
	await ctx.answerCallbackQuery({ text: `Основная модель: ${newMainModel}` })
	await ctx.editMessageText(
		`Основная модель установлена: <b>${newMainModel}</b>`,
		{
			parse_mode: 'HTML',
		}
	)
})

bot.callbackQuery('set_main_model_deepseek', async (ctx) => {
	const userId = ctx.from.id.toString()
	const session = globalContext[userId]
	const newMainModel = 'deepseek-chat'
	const currentBackup = session.backupModel

	if (newMainModel === currentBackup) {
		await ctx.answerCallbackQuery({
			text: 'Нельзя выбрать одинаковую модель как основную и резервную.',
		})
		return
	}

	session.mainModel = newMainModel
	await saveContext(globalContext)
	await ctx.answerCallbackQuery({ text: `Основная модель: ${newMainModel}` })
	await ctx.editMessageText(
		`Основная модель установлена: <b>${newMainModel}</b>`,
		{
			parse_mode: 'HTML',
		}
	)
})

// Функции установки резервной модели
bot.callbackQuery('set_backup_model_geminiexp', async (ctx) => {
	const userId = ctx.from.id.toString()
	const session = globalContext[userId]
	const newBackupModel = 'gemini-exp-1206'
	const currentMain = session.mainModel

	if (newBackupModel === currentMain) {
		await ctx.answerCallbackQuery({
			text: 'Нельзя выбрать одинаковую модель как основную и резервную.',
		})
		return
	}

	session.backupModel = newBackupModel
	await saveContext(globalContext)
	await ctx.answerCallbackQuery({ text: `Резервная модель: ${newBackupModel}` })
	await ctx.editMessageText(
		`Резервная модель установлена: <b>${newBackupModel}</b>`,
		{
			parse_mode: 'HTML',
		}
	)
})

bot.callbackQuery('set_backup_model_geminipro', async (ctx) => {
	const userId = ctx.from.id.toString()
	const session = globalContext[userId]
	const newBackupModel = 'gemini-1.5-pro-002'
	const currentMain = session.mainModel

	if (newBackupModel === currentMain) {
		await ctx.answerCallbackQuery({
			text: 'Нельзя выбрать одинаковую модель как основную и резервную.',
		})
		return
	}

	session.backupModel = newBackupModel
	await saveContext(globalContext)
	await ctx.answerCallbackQuery({ text: `Резервная модель: ${newBackupModel}` })
	await ctx.editMessageText(
		`Резервная модель установлена: <b>${newBackupModel}</b>`,
		{
			parse_mode: 'HTML',
		}
	)
})

bot.callbackQuery('set_backup_model_deepseek', async (ctx) => {
	const userId = ctx.from.id.toString()
	const session = globalContext[userId]
	const newBackupModel = 'deepseek-chat'
	const currentMain = session.mainModel

	if (newBackupModel === currentMain) {
		await ctx.answerCallbackQuery({
			text: 'Нельзя выбрать одинаковую модель как основную и резервную.',
		})
		return
	}

	session.backupModel = newBackupModel
	await saveContext(globalContext)
	await ctx.answerCallbackQuery({ text: `Резервная модель: ${newBackupModel}` })
	await ctx.editMessageText(
		`Резервная модель установлена: <b>${newBackupModel}</b>`,
		{
			parse_mode: 'HTML',
		}
	)
})

// Кнопка "назад" из меню моделей
bot.callbackQuery('back_to_settings', async (ctx) => {
	await ctx.answerCallbackQuery()
	await sendSettingsMenu(ctx)
})

async function sendSettingsMenu(ctx) {
	const description = `Вот что я могу для тебя сделать:
📝 <b>Воспоминания</b> — Посмотреть мои записанные воспоминания о тебе.
🤖 <b>Модель</b> — Узнать, какую модель я сейчас использую.
🔄 <b>Обновить</b> — Обновить мои воспоминания.
🗑️ <b>Очистить</b> — Полностью удалить все воспоминания.
❌ <b>Удалить</b> — Удалить часть сообщений из истории.
🧩 <b>Длина</b> — Изменить максимальную длину ответов.
🔥 <b>Температура</b> — Изменить степень креативности ответов.
⚙️ <b>Выбор моделей</b> — Настроить, какую модель использовать как основную и резервную.`

	const keyboard = new InlineKeyboard()
		.text('📝 Воспоминания', 'about_user')
		.text('🤖 Модель', 'about_model')
		.row()
		.text('🔄 Обновить', 'refresh_memories')
		.text('🗑️ Очистить', 'clear_memories')
		.row()
		.text('❌ Удалить', 'delete_messages')
		.row()
		.text('🧩 Длина', 'adjust_max_tokens')
		.text('🔥 Температура', 'adjust_temperature')
		.row()
		.text('⚙️ Выбор моделей', 'model_settings')

	await ctx.editMessageText(description, {
		parse_mode: 'HTML',
		reply_markup: keyboard,
	})
}

// Обработчик для кнопки "Обновить"
bot.callbackQuery('refresh_memories', async (ctx) => {
	try {
		const userId = ctx.from?.id.toString()
		if (!userId) {
			await ctx.answerCallbackQuery({ text: 'Не удалось получить твой id.' })
			return
		}

		const session = globalContext[userId]
		if (!session || !session.history || session.history.length === 0) {
			await ctx.answerCallbackQuery({
				text: 'У тебя нет истории для обобщения.',
			})
			return
		}

		try {
			await ctx.answerCallbackQuery({
				text: 'Начинаю обновление воспоминаний...',
			})
		} catch (error) {
			console.log('Не удалось ответить на callback query, продолжаем...')
		}

		const statusMessage = await ctx.reply(
			'Обновляю воспоминания. Это может занять некоторое время...'
		)
		const typingSimulation = startTypingSimulation(ctx)

		try {
			await generateSummary(session)
			typingSimulation.stop()
			await ctx.api.editMessageText(
				statusMessage.chat.id,
				statusMessage.message_id,
				'Твои воспоминания успешно обновлены.'
			)

			if (session.memories && Object.keys(session.memories).length > 0) {
				let memoriesText = 'Вот твои обновленные воспоминания:\n'
				Object.entries(session.memories).forEach(([date, memory]) => {
					memoriesText += `<b>Воспоминания за ${date}:</b>\n${memory.text}\n`
				})
				await sendLongMessage(ctx, memoriesText)
			} else {
				await ctx.reply('К сожалению, не удалось сгенерировать воспоминания.')
			}
		} catch (error) {
			console.error('Ошибка при обновлении воспоминаний:', error)
			typingSimulation.stop()
			await ctx.api.editMessageText(
				statusMessage.chat.id,
				statusMessage.message_id,
				'Произошла ошибка при обновлении воспоминаний.'
			)
		}
	} catch (error) {
		console.error(
			'Ошибка при обработке запроса на обновление воспоминаний:',
			error
		)
		await ctx.reply(
			'Произошла ошибка при обработке твоего запроса. Попробуй позже.'
		)
	}
})

// Обработчик для кнопки "Очистить воспоминания"
bot.callbackQuery('clear_memories', async (ctx) => {
	try {
		await ctx.answerCallbackQuery()
		const confirmationKeyboard = new InlineKeyboard()
			.text('Да, удалить', 'confirm_clear_memories')
			.text('Отмена', 'cancel_clear_memories')

		await ctx.reply(
			'Это удалит все воспоминания и историю переписки. Уверена, что хочешь всё забыть?',
			{
				reply_markup: confirmationKeyboard,
			}
		)
	} catch (error) {
		console.error(
			'Ошибка при запросе подтверждения очистки воспоминаний:',
			error
		)
		await ctx.answerCallbackQuery({
			text: 'Произошла ошибка. Попробуй позже.',
		})
	}
})

bot.callbackQuery('confirm_clear_memories', async (ctx) => {
	try {
		const userId = ctx.from?.id.toString()
		if (!userId) {
			await ctx.answerCallbackQuery({ text: 'Не удалось получить твой id.' })
			return
		}

		globalContext[userId].history = []
		globalContext[userId].memories = {}
		globalContext[userId].messageCountSinceSummary = 0
		await saveContext(globalContext)

		await ctx.answerCallbackQuery({
			text: 'Все воспоминания и история удалены.',
		})
		await sendLongMessage(
			ctx,
			'Я очистила все воспоминания и историю нашего общения. Мы можем начать заново!'
		)
	} catch (error) {
		console.error('Ошибка при очистке воспоминаний:', error)
		await ctx.answerCallbackQuery({
			text: 'Произошла ошибка. Попробуй позже.',
		})
	}
})

bot.callbackQuery('cancel_clear_memories', async (ctx) => {
	try {
		await ctx.answerCallbackQuery({ text: 'Очистка воспоминаний отменена.' })
		await sendLongMessage(
			ctx,
			'Хорошо, я продолжаю помнить всё, что было между нами.'
		)
	} catch (error) {
		console.error('Ошибка при отмене очистки воспоминаний:', error)
		await ctx.answerCallbackQuery({
			text: 'Произошла ошибка. Попробуй позже.',
		})
	}
})

bot.callbackQuery('delete_messages', async (ctx) => {
	const keyboard = new InlineKeyboard()
		.text('🗑️ Удалить все сообщения за сегодня', 'delete_today_messages')
		.row()
		.text('📝 Указать количество сообщений', 'delete_specific_number')
		.row()
		.text('⬅️ Назад', 'back_to_settings')

	await ctx.answerCallbackQuery()
	await ctx.editMessageText('Выбери опцию удаления сообщений:', {
		reply_markup: keyboard,
	})
})

bot.callbackQuery('delete_today_messages', async (ctx) => {
	try {
		const userId = ctx.from.id.toString()
		const session = globalContext[userId]

		if (!session || !session.history || session.history.length === 0) {
			await ctx.answerCallbackQuery({ text: 'У тебя нет истории сообщений.' })
			return
		}

		const today = new Date().toLocaleDateString('ru-RU')
		session.history = session.history.filter((msg) => {
			const [msgDate] = msg.date.split(',')
			return msgDate !== today
		})
		await saveContext(globalContext)

		await ctx.answerCallbackQuery({ text: 'Сообщения за сегодня удалены.' })
		await sendLongMessage(ctx, 'Я удалила все сообщения за сегодня.')
	} catch (error) {
		console.error('Ошибка при удалении сообщений за сегодня:', error)
		await ctx.answerCallbackQuery({ text: 'Произошла ошибка.' })
	}
})

bot.callbackQuery('delete_specific_number', async (ctx) => {
	const userId = ctx.from.id.toString()
	globalContext[userId].awaitingMessageDeletionCount = true
	await ctx.answerCallbackQuery()
	await ctx.editMessageText(
		'Введи количество последних сообщений, которое ты хочешь удалить:'
	)
})

// Установка длины
bot.callbackQuery('adjust_max_tokens', async (ctx) => {
	const userId = ctx.from.id.toString()
	const currentValue = globalContext[userId]?.maxOutputTokens || 700

	const keyboard = new InlineKeyboard()
	for (let i = 100; i <= 1000; i += 100) {
		keyboard.text(i.toString(), `set_max_tokens_${i}`)
		if (i % 300 === 0) keyboard.row()
	}
	keyboard.row().text('⬅️ Назад', 'back_to_settings')

	const description =
		'Длина ответа — максимальное число токенов. Чем больше значение, тем длиннее ответ, но дольше ожидание.'
	await ctx.answerCallbackQuery()
	await ctx.editMessageText(
		`${description}\nТекущая максимальная длина ответа: ${currentValue} токенов. Выбери новое значение:`,
		{ reply_markup: keyboard }
	)
})

// Установка температуры
bot.callbackQuery('adjust_temperature', async (ctx) => {
	const userId = ctx.from.id.toString()
	const currentValue = globalContext[userId]?.temperature || 1.5

	const keyboard = new InlineKeyboard()
	for (let i = 0.1; i <= 2.0; i += 0.1) {
		keyboard.text(i.toFixed(1), `set_temperature_${i.toFixed(1)}`)
		if (Math.round(i * 10) % 3 === 0) keyboard.row()
	}
	keyboard.row().text('⬅️ Назад', 'back_to_settings')

	const description =
		'Температура влияет на креативность ответов: низкая — логичные ответы, высокая — творческие.'
	await ctx.answerCallbackQuery()
	await ctx.editMessageText(
		`${description}\nТекущая температура: ${currentValue}. Выбери новое значение:`,
		{ reply_markup: keyboard }
	)
})

// Применение нового количества токенов
bot.callbackQuery(/^set_max_tokens_/, async (ctx) => {
	const userId = ctx.from.id.toString()
	const newValue = parseInt(ctx.callbackQuery.data.split('_').pop())

	if (!globalContext[userId]) {
		globalContext[userId] = {}
	}
	globalContext[userId].maxOutputTokens = newValue
	await saveContext(globalContext)

	await ctx.answerCallbackQuery({
		text: `Максимальная длина ответа: ${newValue} токенов.`,
	})
	await ctx.editMessageText(
		`Новая максимальная длина ответа: ${newValue} токенов.`
	)
})

// Применение новой температуры
bot.callbackQuery(/^set_temperature_/, async (ctx) => {
	const userId = ctx.from.id.toString()
	const newValue = parseFloat(ctx.callbackQuery.data.split('_').pop())

	if (newValue < 0 || newValue > 2) {
		await ctx.answerCallbackQuery({
			text: 'Пожалуйста, выбери значение температуры между 0 и 2.',
		})
		return
	}

	if (!globalContext[userId]) {
		globalContext[userId] = {}
	}
	globalContext[userId].temperature = newValue
	await saveContext(globalContext)

	await ctx.answerCallbackQuery({ text: `Температура: ${newValue}.` })
	await ctx.editMessageText(`Новая температура: ${newValue}.`)
})

function groupMessagesByDay(messages) {
	const groupedMessages = {}
	messages.forEach((msg) => {
		const date = msg.date.split(',')[0]
		if (!groupedMessages[date]) {
			groupedMessages[date] = []
		}
		groupedMessages[date].push(msg)
	})
	return groupedMessages
}

async function generateResponseWithRetry(
	model,
	contents,
	retries = 1,
	initialDelay = 1000
) {
	let delay = initialDelay
	for (let attempt = 1; attempt <= retries; attempt++) {
		try {
			return await model.generateContent(contents)
		} catch (error) {
			if ((error.status === 503 || error.status === 429) && attempt < retries) {
				console.log(
					`Попытка ${attempt} не удалась (${error.status}). Повтор через ${
						delay / 1000
					}с...`
				)
				await new Promise((resolve) => setTimeout(resolve, delay))
				delay += 1000
			} else {
				throw error
			}
		}
	}
}

async function generateSummary(session) {
	try {
		const history = session.history
		if (!history || history.length === 0) {
			console.log('История сообщений пуста.')
			return
		}

		const groupedMessages = groupMessagesByDay(history)
		const today = new Date().toLocaleDateString('ru-RU')
		const memories = session.memories || {}

		for (const [date, messages] of Object.entries(groupedMessages)) {
			let previousMemoriesText = ''
			for (const [prevDate, memory] of Object.entries(memories)) {
				if (prevDate !== date) {
					previousMemoriesText += `Воспоминания за ${prevDate}:\n${memory.text}\n`
				}
			}

			if (
				!memories[date] ||
				memories[date].text.trim().length === 0 ||
				date === today
			) {
				const historyText = messages
					.map((msg) => `${msg.role} (${msg.date}): ${msg.content}`)
					.join('\n')

				const prompt = `Ты Гермиона Грейнджер и твоя задача, посмотрев на эту переписку за ${date}, ... (твой актуальный запрос на суммаризацию)`

				const contents = [
					{ text: `Предыдущие воспоминания:\n${previousMemoriesText}` },
					{ text: `${prompt}\n${historyText}` },
				]

				const result = await generateResponseWithRetry(
					summarizationModel,
					contents
				)
				if (!result.response) {
					throw new Error(`Не удалось сгенерировать суммаризацию за ${date}`)
				}

				let summary = result.response.text()
				summary = sanitizeHtml(summary, {
					allowedTags: allowedTags,
					allowedAttributes: allowedAttributes,
					allowedClasses: {
						span: ['tg-spoiler'],
						code: ['language-python'],
						pre: ['language-python'],
						blockquote: ['expandable'],
					},
					allowedSchemes: ['http', 'https', 'tg'],
					allowedSchemesByTag: {
						a: ['http', 'https', 'tg'],
					},
				})

				memories[date] = {
					text: summary,
					date: new Date().toLocaleString(),
				}
			}
		}

		session.memories = memories
		session.history = groupedMessages[today] || []
		await saveContext(globalContext)

		console.log('Суммаризация успешно сгенерирована.')
	} catch (error) {
		console.error('Ошибка при автоматическом обновлении воспоминаний:', error)
	}
}

const summarizationModel = createGeminiModel(
	'gemini-1.5-flash-8b-001',
	1500,
	0.5
)

async function generateGeminiResponse(model, contents) {
	try {
		const result = await generateResponseWithRetry(model, contents, 3, 1000)
		return result
	} catch (error) {
		if (error.status === 503) {
			throw new Error('Модель временно недоступна. Попробуй позже.')
		} else if (error.status === 429) {
			throw new Error('Слишком много запросов. Попробуй позже.')
		} else if (
			error.response &&
			error.response.promptFeedback &&
			error.response.promptFeedback.blockReason === 'PROHIBITED_CONTENT'
		) {
			throw new Error('Ответ заблокирован из-за запрещённого содержания.')
		} else {
			throw error
		}
	}
}

async function generateResponseWithBackup(session, contents) {
	const mainModelName = session.mainModel || 'gemini-exp-1206'
	const backupModelName = session.backupModel || 'gemini-1.5-pro-002'
	const maxOutputTokens = session.maxOutputTokens || 700
	const temperature = session.temperature || 1.5

	let mainModel
	let backupModel

	if (mainModelName === 'deepseek-chat') {
		mainModel = createDeepseekModel(maxOutputTokens, temperature)
	} else {
		mainModel = createGeminiModel(mainModelName, maxOutputTokens, temperature)
	}

	if (backupModelName === 'deepseek-chat') {
		backupModel = createDeepseekModel(maxOutputTokens, temperature)
	} else {
		backupModel = createGeminiModel(
			backupModelName,
			maxOutputTokens,
			temperature
		)
	}

	try {
		const result = await generateGeminiResponse(mainModel, contents)
		return result
	} catch (error) {
		console.error('Ошибка при основной модели:', error)
		console.log('Пробуем резервную...')
		try {
			const result = await generateGeminiResponse(backupModel, contents)
			result.usedBackupModel = true
			return result
		} catch (backupError) {
			console.error('Ошибка при резервной модели:', backupError)
			throw error
		} finally {
			cleanupModel(backupModel)
		}
	} finally {
		cleanupModel(mainModel)
	}
}

function isMessageFromToday(messageDateString) {
	const today = new Date()
	const [datePart] = messageDateString.split(',')
	const [day, month, year] = datePart.split('.').map(Number)

	return (
		day === today.getDate() &&
		month === today.getMonth() + 1 &&
		year === today.getFullYear()
	)
}

// Обработка входящих текстовых сообщений
bot.chatType('private').on('message:text', async (ctx) => {
	const userId = ctx.from.id.toString()

	if (globalContext[userId]?.awaitingMessageDeletionCount) {
		const input = ctx.message.text.trim()
		const number = parseInt(input, 10)

		if (isNaN(number) || number <= 0) {
			await ctx.reply('Введи корректное положительное число.')
		} else {
			const session = globalContext[userId]
			if (!session || !session.history || session.history.length === 0) {
				await ctx.reply('У тебя нет истории сообщений.')
			} else {
				session.history.splice(-number)
				await saveContext(globalContext)
				await ctx.reply(`Я удалила последние ${number} сообщений из истории.`)
			}
		}

		globalContext[userId].awaitingMessageDeletionCount = false
		return
	}

	let userMessage = ctx.message.text
	const session = globalContext[userId]

	try {
		const userName = getUserName(ctx)
		const messageDate = new Date(ctx.message.date * 1000).toLocaleString()

		const history = session.history
		const memories = session.memories || {}

		const contents = buildContents(
			history,
			memories,
			userName,
			userMessage,
			messageDate
		)
		const typingSimulation = startTypingSimulation(ctx)

		let result
		try {
			result = await generateResponseWithBackup(session, contents)
		} finally {
			typingSimulation.stop()
		}

		await new Promise((resolve) => setTimeout(resolve, 1000))
		if (!result.response) {
			throw new Error('Не удалось сгенерировать ответ')
		}

		let botReply = result.response.text()
		if (result.usedBackupModel) {
			botReply = '<i>резервная модель:</i>\n\n' + botReply
		}

		botReply = sanitizeHtml(botReply, {
			allowedTags: allowedTags,
			allowedAttributes: allowedAttributes,
			allowedClasses: {
				span: ['tg-spoiler'],
				code: ['language-python'],
				pre: ['language-python'],
				blockquote: ['expandable'],
			},
			allowedSchemes: ['http', 'https', 'tg'],
			allowedSchemesByTag: {
				a: ['http', 'https', 'tg'],
			},
		})

		if (!botReply || botReply.trim() === '') {
			botReply = 'Извини, но я не смогла сформулировать ответ.'
		}

		session.history.push({
			role: userName,
			content: userMessage,
			date: messageDate,
		})
		session.history.push({
			role: 'Гермиона',
			content: botReply,
			date: new Date().toLocaleString(),
		})

		session.messageCountSinceSummary =
			(session.messageCountSinceSummary || 0) + 2
		if (session.messageCountSinceSummary >= 30) {
			await generateSummary(session)
			session.messageCountSinceSummary = 0
		}

		await sendLongMessage(ctx, botReply)
	} catch (error) {
		console.error('Ошибка при обработке текстового сообщения:', error)
		let errorMessage = 'Произошла ошибка. Попробуй чуть позже.'
		if (error.message.includes('PROHIBITED_CONTENT')) {
			errorMessage = 'Извини, твой запрос содержит запрещённый контент.'
		} else if (error.message.includes('недоступна')) {
			errorMessage = 'Извини, сервис временно недоступен.'
		} else if (error.message.includes('Слишком много запросов')) {
			errorMessage = 'Извини, слишком много запросов. Попробуй позже.'
		}
		await ctx.reply(errorMessage)
	}
})

// Обработка изображений (по-прежнему на Gemini)
bot.on('message:photo', async (ctx) => {
	let userModel = null
	try {
		const chatType = ctx.chat.type

		if (chatType === 'group' || chatType === 'supergroup') {
			if (!isAllowedGroup(ctx.chat.id)) {
				await ctx.reply(
					'Извини, я не могу общаться в этой группе. Напиши мне в личку или в @AIKaleidoscope.',
					{ reply_to_message_id: ctx.message.message_id }
				)
				return
			}
			if (!botInfo) {
				botInfo = await bot.api.getMe()
			}
			const botId = botInfo.id
			const botUsername = `@${botInfo.username.toLowerCase()}`

			const captionEntities = ctx.message.caption_entities || []
			const isMentioned = captionEntities.some((entity) => {
				if (entity.type === 'mention' && ctx.message.caption) {
					const mention = ctx.message.caption
						.substring(entity.offset, entity.offset + entity.length)
						.toLowerCase()
					return mention === botUsername.toLowerCase()
				}
				return false
			})

			const isReplyToBot =
				ctx.message.reply_to_message &&
				ctx.message.reply_to_message.from &&
				ctx.message.reply_to_message.from.id === botId

			if (!isMentioned && !isReplyToBot) {
				return
			}
		}

		const photos = ctx.message.photo
		const caption = ctx.message.caption || ''
		const userId = ctx.from.id.toString()
		if (!photos || photos.length === 0) {
			await ctx.reply('Не удалось получить изображение.')
			return
		}

		const userMaxTokens = globalContext[userId]?.maxOutputTokens || 700
		const userTemperature = globalContext[userId]?.temperature || 1.5
		userModel = createGeminiModel(
			'gemini-1.5-pro-002',
			userMaxTokens,
			userTemperature
		)

		const highestResPhoto = photos[photos.length - 1]
		const fileId = highestResPhoto.file_id

		await new Promise((resolve) => setTimeout(resolve, 1000))
		const localFilePath = await downloadTelegramFile(fileId)

		let mimeType = 'image/jpeg'
		const fileExtension = path.extname(localFilePath).toLowerCase()
		if (fileExtension === '.png') {
			mimeType = 'image/png'
		} else if (fileExtension === '.webp') {
			mimeType = 'image/webp'
		} else if (fileExtension === '.heic') {
			mimeType = 'image/heic'
		} else if (fileExtension === '.heif') {
			mimeType = 'image/heif'
		}

		const supportedMimeTypes = [
			'image/jpeg',
			'image/png',
			'image/webp',
			'image/heic',
			'image/heif',
		]
		if (!supportedMimeTypes.includes(mimeType)) {
			await ctx.reply('Извини, этот тип файла не поддерживается.')
			return
		}

		const displayName = `User Image ${Date.now()}`
		const fileUri = await uploadFileToGemini(
			localFilePath,
			mimeType,
			displayName
		)
		await fs.unlink(localFilePath)

		let prompt
		if (caption.trim().length > 0) {
			prompt = `${sanitizeHtml(caption.trim(), {
				allowedTags: [],
				allowedAttributes: {},
			})}`
		} else {
			prompt = 'Опиши содержимое этого изображения.'
		}

		const userName = getUserName(ctx)
		const messageDate = new Date(ctx.message.date * 1000).toLocaleString()
		const session = globalContext[userId]
		const history = session.history
		const memories = session.memories || {}

		const contents = buildContents(
			history,
			memories,
			userName,
			prompt,
			messageDate
		)
		contents.push({
			fileData: {
				mimeType: mimeType,
				fileUri: fileUri,
			},
		})

		const typingSimulation = startTypingSimulation(ctx)
		let result
		try {
			result = await generateGeminiResponse(userModel, contents)
		} finally {
			typingSimulation.stop()
		}

		await new Promise((resolve) => setTimeout(resolve, 1000))
		if (!result.response) {
			throw new Error('Не удалось сгенерировать ответ')
		}

		let botReply = result.response.text()
		botReply = sanitizeHtml(botReply, {
			allowedTags: allowedTags,
			allowedAttributes: allowedAttributes,
			allowedClasses: {
				span: ['tg-spoiler'],
				code: ['language-python'],
				pre: ['language-python'],
				blockquote: ['expandable'],
			},
			allowedSchemes: ['http', 'https', 'tg'],
			allowedSchemesByTag: {
				a: ['http', 'https', 'tg'],
			},
		})

		if (!botReply || botReply.trim() === '') {
			botReply = 'Извини, но я не смогла обработать это изображение.'
		}

		session.history.push({
			role: userName,
			content: caption
				? `Отправил(а) изображение с комментарием: "${caption}"`
				: 'Отправил(а) изображение',
			date: messageDate,
		})
		session.history.push({
			role: 'Гермиона',
			content: botReply,
			date: new Date().toLocaleString(),
		})

		session.messageCountSinceSummary =
			(session.messageCountSinceSummary || 0) + 2
		if (session.messageCountSinceSummary >= 30) {
			await generateSummary(session)
			session.messageCountSinceSummary = 0
		}

		let replyOptions = {}
		if (chatType === 'group' || chatType === 'supergroup') {
			replyOptions.reply_to_message_id = ctx.message.message_id
		}

		await sendLongMessage(ctx, botReply, replyOptions)
	} catch (error) {
		console.error('Ошибка при обработке изображения:', error)
		let errorMessage = 'Произошла ошибка. Попробуй позже.'
		if (error.message.includes('PROHIBITED_CONTENT')) {
			errorMessage = 'Извини, твой запрос содержит запрещённый контент.'
		} else if (error.message.includes('недоступна')) {
			errorMessage = 'Извини, сервис временно недоступен.'
		} else if (error.message.includes('Слишком много запросов')) {
			errorMessage = 'Извини, слишком много запросов. Попробуй позже.'
		}
		await ctx.reply(errorMessage, {
			reply_to_message_id: ctx.message.message_id,
		})
	} finally {
		if (userModel) {
			cleanupModel(userModel)
		}
	}
})

// Обработка аудио/войса (теперь принудительно на gemini-1.5-pro-002)
bot.on(['message:voice', 'message:audio'], async (ctx) => {
	let userModel = null
	try {
		const chatType = ctx.chat.type

		if (chatType === 'group' || chatType === 'supergroup') {
			if (!isAllowedGroup(ctx.chat.id)) {
				await ctx.reply(
					'Извини, я не могу общаться в этой группе. Напиши мне в личку или в @AIKaleidoscope.',
					{ reply_to_message_id: ctx.message.message_id }
				)
				return
			}

			if (!botInfo) {
				botInfo = await bot.api.getMe()
			}
			const botId = botInfo.id
			const botUsername = `@${botInfo.username.toLowerCase()}`

			const captionEntities = ctx.message.caption_entities || []
			const isMentioned = captionEntities.some((entity) => {
				if (entity.type === 'mention' && ctx.message.caption) {
					const mention = ctx.message.caption
						.substring(entity.offset, entity.offset + entity.length)
						.toLowerCase()
					return mention === botUsername.toLowerCase()
				}
				return false
			})

			const isReplyToBot =
				ctx.message.reply_to_message &&
				ctx.message.reply_to_message.from &&
				ctx.message.reply_to_message.from.id === botId

			if (!isMentioned && !isReplyToBot) {
				return
			}
		}

		let fileId
		let mimeType
		const userId = ctx.from.id.toString()

		if (ctx.message.voice) {
			fileId = ctx.message.voice.file_id
			mimeType = 'audio/ogg'
		} else if (ctx.message.audio) {
			fileId = ctx.message.audio.file_id
			mimeType = ctx.message.audio.mime_type || 'audio/mpeg'
		}

		if (!fileId) {
			await ctx.reply('Не удалось получить аудиосообщение.')
			return
		}

		const supportedMimeTypes = [
			'audio/ogg',
			'audio/mpeg',
			'audio/wav',
			'audio/mp3',
			'audio/aac',
			'audio/flac',
		]
		if (!supportedMimeTypes.includes(mimeType)) {
			await ctx.reply('Извини, этот тип аудиофайла не поддерживается.')
			return
		}

		// Именно здесь используем gemini-1.5-pro-002
		const userMaxTokens = globalContext[userId]?.maxOutputTokens || 700
		const userTemperature = globalContext[userId]?.temperature || 1.5
		userModel = createGeminiModel(
			'gemini-1.5-pro-002',
			userMaxTokens,
			userTemperature
		)

		await new Promise((resolve) => setTimeout(resolve, 1000))
		const localFilePath = await downloadTelegramFile(fileId)

		const displayName = `User Audio ${Date.now()}`
		const fileUri = await uploadFileToGemini(
			localFilePath,
			mimeType,
			displayName
		)
		await fs.unlink(localFilePath)

		const userName = getUserName(ctx)
		const messageDate = new Date(ctx.message.date * 1000).toLocaleString()
		const session = globalContext[userId]
		const history = session.history
		const memories = session.memories || {}

		const contents = buildContents(
			history,
			memories,
			userName,
			'Отправил(а) аудиосообщение.',
			messageDate
		)
		contents.push({
			fileData: {
				mimeType: mimeType,
				fileUri: fileUri,
			},
		})

		const typingSimulation = startTypingSimulation(ctx)
		let result
		try {
			result = await generateGeminiResponse(userModel, contents)
		} finally {
			typingSimulation.stop()
		}

		await new Promise((resolve) => setTimeout(resolve, 1000))
		if (!result.response) {
			throw new Error('Не удалось сгенерировать ответ')
		}
		let botReply = result.response.text()

		botReply = sanitizeHtml(botReply, {
			allowedTags: allowedTags,
			allowedAttributes: allowedAttributes,
			allowedClasses: {
				span: ['tg-spoiler'],
				code: ['language-python'],
				pre: ['language-python'],
				blockquote: ['expandable'],
			},
			allowedSchemes: ['http', 'https', 'tg'],
			allowedSchemesByTag: {
				a: ['http', 'https', 'tg'],
			},
		})

		if (!botReply || botReply.trim() === '') {
			botReply = 'Извини, но я не смогла обработать это аудиосообщение.'
		}

		session.history.push({
			role: userName,
			content: 'Отправил(а) аудиосообщение.',
			date: messageDate,
		})
		session.history.push({
			role: 'Гермиона',
			content: botReply,
			date: new Date().toLocaleString(),
		})

		session.messageCountSinceSummary =
			(session.messageCountSinceSummary || 0) + 2
		if (session.messageCountSinceSummary >= 30) {
			await generateSummary(session)
			session.messageCountSinceSummary = 0
		}

		let replyOptions = {}
		if (chatType === 'group' || chatType === 'supergroup') {
			replyOptions.reply_to_message_id = ctx.message.message_id
		}

		await sendLongMessage(ctx, botReply, replyOptions)
	} catch (error) {
		console.error('Ошибка при обработке аудиосообщения:', error)
		let errorMessage =
			'Произошла ошибка при обработке твоего аудио. Попробуй позже.'
		if (error.message.includes('PROHIBITED_CONTENT')) {
			errorMessage = 'Извини, твой запрос содержит запрещённый контент.'
		} else if (error.message.includes('недоступна')) {
			errorMessage = 'Извини, сервис временно недоступен.'
		} else if (error.message.includes('Слишком много запросов')) {
			errorMessage = 'Извини, слишком много запросов. Попробуй позже.'
		}
		await ctx.reply(errorMessage, {
			reply_to_message_id: ctx.message.message_id,
		})
	} finally {
		if (userModel) {
			cleanupModel(userModel)
		}
	}
})

// Обработка сообщений в группах
bot.chatType(['group', 'supergroup']).on('message', async (ctx) => {
	let userModel = null
	try {
		const chat = ctx.chat
		if (!chat) {
			return
		}

		if (!botInfo) {
			try {
				botInfo = await bot.api.getMe()
			} catch (error) {
				console.error('Ошибка при получении инфо о боте:', error)
				return
			}
		}

		const botUsername = `@${botInfo.username}`
		const entities = ctx.message.entities || []
		const isMentioned = entities.some((entity) => {
			return (
				entity.type === 'mention' &&
				ctx.message.text &&
				ctx.message.text
					.substring(entity.offset, entity.offset + entity.length)
					.toLowerCase() === botUsername.toLowerCase()
			)
		})

		const isReply =
			ctx.message.reply_to_message &&
			ctx.message.reply_to_message.from &&
			ctx.message.reply_to_message.from.id === botInfo.id

		if (!isMentioned && !isReply) {
			return
		}

		if (!isAllowedGroup(chat.id)) {
			await ctx.reply(
				'Извини, я не могу общаться в этой группе, но ты можешь пообщаться со мной в личных сообщениях или в @AIKaleidoscope.',
				{ reply_to_message_id: ctx.message.message_id }
			)
			console.log('Chat ID:', chat.id)
			return
		}

		await new Promise((resolve) => setTimeout(resolve, 1000))

		let userMessage = ''
		if (isMentioned && ctx.message.text) {
			userMessage = ctx.message.text
				.replace(new RegExp(botUsername, 'gi'), '')
				.trim()
		} else if (isReply && ctx.message.text) {
			userMessage = ctx.message.text.trim()
		}

		const userId = ctx.from.id.toString()
		const session = globalContext[userId]
		const userName = getUserName(ctx)
		const messageDate = new Date(ctx.message.date * 1000).toLocaleString()
		const history = session.history
		const memories = session.memories || {}
		const contents = buildContents(
			history,
			memories,
			userName,
			userMessage,
			messageDate
		)

		const typingSimulation = startTypingSimulation(ctx)
		let result
		try {
			result = await generateResponseWithBackup(session, contents)
		} finally {
			typingSimulation.stop()
		}

		await new Promise((resolve) => setTimeout(resolve, 1000))
		if (!result.response) {
			throw new Error('Не удалось сгенерировать ответ')
		}

		let botReply = result.response.text()
		if (result.usedBackupModel) {
			botReply = '<i>резервная модель:</i>\n\n' + botReply
		}

		botReply = sanitizeHtml(botReply, {
			allowedTags: allowedTags,
			allowedAttributes: allowedAttributes,
			allowedClasses: {
				span: ['tg-spoiler'],
				code: ['language-python'],
				pre: ['language-python'],
				blockquote: ['expandable'],
			},
			allowedSchemes: ['http', 'https', 'tg'],
			allowedSchemesByTag: {
				a: ['http', 'https', 'tg'],
			},
		})

		if (!botReply || botReply.trim() === '') {
			botReply = 'Извини, но я не смогла сформулировать ответ.'
		}

		session.history.push({
			role: userName,
			content: userMessage,
			date: messageDate,
		})
		session.history.push({
			role: 'Гермиона',
			content: botReply,
			date: new Date().toLocaleString(),
		})

		session.messageCountSinceSummary =
			(session.messageCountSinceSummary || 0) + 2
		if (session.messageCountSinceSummary >= 30) {
			await generateSummary(session)
			session.messageCountSinceSummary = 0
		}

		await sendLongMessage(ctx, botReply, {
			reply_to_message_id: ctx.message.message_id,
		})
	} catch (error) {
		console.error('Ошибка при обработке сообщения в группе:', error)
		let errorMessage = 'Произошла ошибка. Попробуй чуть позже.'
		if (error.message.includes('PROHIBITED_CONTENT')) {
			errorMessage = 'Извини, твой запрос содержит запрещённый контент.'
		} else if (error.message.includes('недоступна')) {
			errorMessage = 'Извини, сервис временно недоступен.'
		} else if (error.message.includes('Слишком много запросов')) {
			errorMessage = 'Извини, слишком много запросов. Попробуй позже.'
		}
		await ctx.reply(errorMessage, {
			reply_to_message_id: ctx.message.message_id,
		})
	} finally {
		if (userModel) {
			cleanupModel(userModel)
		}
	}
})

// Обработка ошибок
bot.catch((err) => {
	const ctx = err.ctx
	console.error(`Ошибка при обработке обновления ${ctx.update.update_id}:`)
	const e = err.error
	if (e instanceof GrammyError) {
		console.error('Ошибка в запросе:', e.description)
	} else if (e instanceof HttpError) {
		console.error('Не удалось связаться с Telegram:', e)
	} else {
		console.error('Неизвестная ошибка:', e)
	}
})

/**
 * Функция для инициализации контекста при запуске
 */
async function initializeContext() {
	try {
		globalContext = await loadContext()
		console.log('Контекст успешно загружен из файла.')
	} catch (error) {
		console.error('Ошибка при загрузке контекста:', error)
		globalContext = {}
	}

	for (let userId in globalContext) {
		if (!globalContext[userId].maxOutputTokens) {
			globalContext[userId].maxOutputTokens = 700
		}
		if (!globalContext[userId].temperature) {
			globalContext[userId].temperature = 1.5
		}
		if (!globalContext[userId].mainModel) {
			globalContext[userId].mainModel = 'gemini-exp-1206'
		}
		if (!globalContext[userId].backupModel) {
			globalContext[userId].backupModel = 'gemini-1.5-pro-002'
		}
	}

	await saveContext(globalContext)
}

/**
 * Функция запуска бота
 */
async function startBot() {
	try {
		await initializeContext()
		botInfo = await bot.api.getMe()
		console.log(`Бот запущен: @${botInfo.username}`)
		await bot.start()
	} catch (error) {
		console.error('Ошибка при запуске бота:', error)
		process.exit(1)
	}
}

// Запуск бота
startBot()
